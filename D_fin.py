
# Запуск решения осуществлялся через Google Colab на T4
# Импортируем библиотеки
import pandas as pd
import numpy as np
import zipfile
from scipy.sparse.csgraph import floyd_warshall
from scipy.optimize import linear_sum_assignment
from geopy.geocoders import Nominatim
from tqdm.auto import tqdm
import json
import os
import time
import requests 


zip_path = "D_data/NTO_BDML_2E_leak.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

COORDS_CACHE_FILE = 'coords.json'
ROAD_DIST_CACHE_FILE = 'road_distances.json'


def get_coordinates(city_names):
    if os.path.exists(COORDS_CACHE_FILE) and os.path.getsize(COORDS_CACHE_FILE) > 0:
        with open(COORDS_CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if all(city in data for city in city_names):
                return data

    print("Сбор координат городов")
    geolocator = Nominatim(user_agent="city_matcher_app_v9")
    coords = {}
    for city in tqdm(city_names):
        location = geolocator.geocode(f"{city}, Россия", timeout=10)
        coords[city] = {"lon": location.longitude, "lat": location.latitude} if location else None
        time.sleep(1.1)

    with open(COORDS_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(coords, f, ensure_ascii=False, indent=4)
    return coords


def get_road_distance_osrm(coord1, coord2):
    url = f"http://router.project-osrm.org/route/v1/driving/{coord1['lon']},{coord1['lat']};{coord2['lon']},{coord2['lat']}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data['code'] == 'Ok':
            distance_in_meters = data['routes'][0]['distance']
            distance_in_km = distance_in_meters / 1000
            return distance_in_km
    return None


def calculate_road_distance_matrix(city_names, coords):
    num_cities = len(city_names)

    if os.path.exists(ROAD_DIST_CACHE_FILE):
        with open(ROAD_DIST_CACHE_FILE, 'r') as f:
            cached = json.load(f)
            if len(cached) == num_cities and len(cached[0]) == num_cities:
                print("Загружена кэшированная матрица дорожных расстояний")
                return np.arr(cached)

    print("Расчет матрицы дорожных расстояний через OSRM(открытый сайт, вычисляющий расстояния по дорогам между городами)")
    dist_matrix = np.zeros((num_cities, num_cities))

    coords_list = [coords[name] for name in city_names]

    coords_str = ";".join([f"{c['lon']},{c['lat']}" for c in coords_list])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}"

    response = requests.get(url, timeout=120)

    if response.status_code == 200:
        data = response.json()
        if data['code'] == 'Ok':
            durations = np.arr(data['durations'])

            average_speed_km_per_hour = 60
            durations_in_hours = durations / 3600
            dist_matrix = durations_in_hours * average_speed_km_per_hour

            print("Получена матрица времени, переводим в расстояния")

    if dist_matrix.sum() == 0:
        print("Используем попарные запросы")
        for i in tqdm(range(num_cities)):
            for j in range(i+1, num_cities):
                dist = get_road_distance_osrm(coords_list[i], coords_list[j])
                if dist:
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                time.sleep(0.5)

    with open(ROAD_DIST_CACHE_FILE, 'w') as f:
        json.dump(dist_matrix.tolist(), f)

    return dist_matrix


def calculate_geodesic_matrix(city_names, coords):
    num_cities = len(city_names)
    coords_arr = np.arr([[coords[name]['lon'], coords[name]['lat']]
                             for name in city_names])

    lon = coords_arr[:, 0]
    lat = coords_arr[:, 1]

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    delta_lon = lon_rad[:, np.newaxis] - lon_rad
    delta_lat = lat_rad[:, np.newaxis] - lat_rad

    a = (np.sin(delta_lat / 2.0) ** 2 +
         np.cos(lat_rad)[:, np.newaxis] * np.cos(lat_rad) *
         np.sin(delta_lon / 2.0) ** 2)

    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0
    distance_matrix = earth_radius_km * c

    return distance_matrix


def get_graph_features(dist_matrix):
    n = len(dist_matrix)
    features = []

    for i in range(n):
        row = dist_matrix[i]
        sorted_row = np.sort(row)

        distances_to_other_cities = sorted_row[1:]

        feat = {
            'min_dist': sorted_row[1] if n > 1 else 0,
            'max_dist': sorted_row[-1],
            'mean_dist': np.mean(distances_to_other_cities),
            'median_dist': np.median(distances_to_other_cities),
            'std_dist': np.std(distances_to_other_cities),
            'sum_dist': np.sum(distances_to_other_cities),
            'nearest_3': np.sum(sorted_row[1:4]),
            'nearest_5': np.sum(sorted_row[1:6]),
            'nearest_10': np.sum(sorted_row[1:11]),
            'q25': np.percentile(distances_to_other_cities, 25),
            'q75': np.percentile(distances_to_other_cities, 75),
        }
        features.append(feat)

    return pd.DataFrame(features).values


def main():
    train_df = pd.read_csv("D_data/dists.csv")
    sample_df = pd.read_csv("D_data/sample_submition.csv")

    city_ids_sorted = sorted(sample_df['city_id'].unique())
    city_names_sorted = sorted(sample_df['city_name'].unique())
    num_cities = len(city_ids_sorted)

    print(f"Количество городов: {num_cities}")

    city_id_to_idx = {name: i for i, name in enumerate(city_ids_sorted)}

    print("Восстановление матрицы расстояний")

    dist_matrix_sparse = np.full((num_cities, num_cities), np.inf)
    np.fill_diagonal(dist_matrix_sparse, 0)

    for _, row in train_df.iterrows():
        i = city_id_to_idx[row['from']]
        j = city_id_to_idx[row['to']]
        dist_matrix_sparse[i, j] = row['dist']
        dist_matrix_sparse[j, i] = row['dist']

    D_given = floyd_warshall(csgraph=dist_matrix_sparse, directed=False)

    finite_distances = D_given[np.isfinite(D_given)]
    max_dist = np.max(finite_distances)
    D_given[np.isinf(D_given)] = max_dist * 2.0

    print("Получение координат")
    real_coords_dict = get_coordinates(city_names_sorted)

    missing = [name for name in city_names_sorted if real_coords_dict.get(name) is None]
    if missing:
        print(f"Не найдены координаты для: {missing}")

    print("Расчет реальных расстояний")

    D_real = calculate_road_distance_matrix(city_names_sorted, real_coords_dict)

    if D_real is None or D_real.sum() == 0:
        print("Не удалось получить дорожные расстояния")
        print("Используем геодезические расстояния с коэффициентом")
        D_real = calculate_geodesic_matrix(city_names_sorted, real_coords_dict)
        road_winding_coefficient = 1.3
        D_real = D_real * road_winding_coefficient

    print("Построение матрицы")

    def get_standardized_signatures(dist_matrix):
        signatures = np.sort(dist_matrix, axis=1)
        mean = signatures.mean(axis=1, keepdims=True)
        std = signatures.std(axis=1, keepdims=True)
        std[std == 0] = 1
        standardized = (signatures - mean) / std
        return standardized

    Sig_given_norm = get_standardized_signatures(D_given)
    Sig_real_norm = get_standardized_signatures(D_real)

    feat_given = get_graph_features(D_given)
    feat_real = get_graph_features(D_real)

    feat_given_mean = feat_given.mean(axis=0)
    feat_given_std = feat_given.std(axis=0) + 1e-8
    feat_given = (feat_given - feat_given_mean) / feat_given_std

    feat_real_mean = feat_real.mean(axis=0)
    feat_real_std = feat_real.std(axis=0) + 1e-8
    feat_real = (feat_real - feat_real_mean) / feat_real_std

    cost_matrix = np.zeros((num_cities, num_cities))

    signature_weight = 0.7
    feature_weight = 0.3

    for i in range(num_cities):
        for j in range(num_cities):
            signature_difference = Sig_given_norm[i] - Sig_real_norm[j]
            sig_cost = np.mean(signature_difference ** 2)

            feature_difference = feat_given[i] - feat_real[j]
            feat_cost = np.mean(feature_difference ** 2)

            cost_matrix[i, j] = signature_weight * sig_cost + feature_weight * feat_cost

    print("Запуск алгоритма")
    given_indices, real_indices = linear_sum_assignment(cost_matrix)

    id_to_name_map = {}
    for given_idx, real_idx in zip(given_indices, real_indices):
        city_id = city_ids_sorted[given_idx]
        city_name = city_names_sorted[real_idx]
        id_to_name_map[city_id] = city_name

    submission_df = pd.DataFrame({
        'city_id': sample_df['city_id'],
        'city_name': sample_df['city_id'].map(id_to_name_map)
    })

    submission_df.to_csv("D_data/submission.csv", index=False)
    print("Решение сохранено")
    print(submission_df.head(10))


if __name__ == '__main__':
    main()