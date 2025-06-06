# Система для определения наличия диабета

## Описание проекта
Цель проекта — построение модели классификации на основе нейронной сети (NN), определяющей наличие диабета на основе медицинских данных.

В результате была создана и обучена нейронная сеть, способная диагностировать заболевание с точностью 81%.

## Авторы
AlyonaHV

Aleksandr62aa

## Исходные данные
Для обучения и тестирования модели использовался датасет из 768 объектов, каждый из которых описан 8 признаками и меткой класса:

0 — диабет не выявлен

1 — диабет выявлен

## Архитектура модели нейронной сети
Количество слоёв: 5

Входной слой: 8 нейронов

Выходной слой: 2 нейрона (по числу классов)

Активация: ReLU

## Установка и запуск проекта (в Google Colab):
Склонируйте репозиторий:
```
!git clone https://github.com/Aleksandr62aa/diabetes_nn_project.git
```
Перейдите в директорию проекта:
```
%cd diabetes_nn_project
```
Установите зависимости:
```
!pip install -r requirements.txt
```
Перед запуском проекта вы можете изменить начальные параметры в конфигурационном файле configs/config.toml.

Далее запустите проект:
```
from main import main
main()
```
## Пример работы кода
Ниже приведён пример запуска алгоритма с начальными параметрами, а также с выводом статистики, архитектуры модели и результатов её обучения:

![1](https://github.com/Aleksandr62aa/diabetes_nn_project/blob/main/results/results.png)

Структура ветвления Git проекта представлена ниже:

```
main
└── prod_docker_version
    └── multicamera
        ├── feature/triton
        └── feature/influx
```


drone_ai_project/
├── README.md
├── requirements.txt
├── environment.yml             # (если используешь conda)
├── yolov8_tracking/            # Детекция + трекинг
│   ├── detect_and_track.py     # основной скрипт YOLOv8 + трекинг
│   ├── config/
│   │   └── bytetrack.yaml      # конфиг для трекера
│   └── weights/
│       └── yolov8n.pt          # веса модели
│
├── dataset/                    # Данные
│   ├── visdrone/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── drone_video.mp4         # Тестовое видео
│   └── visdrone.yaml           # data config для YOLOv8
│
├── training/                   # Обучение моделей
│   ├── train_yolo.py
│   └── logs/
│       └── yolov8_train.log
│
├── inference/                  # Скрипты для инференса
│   ├── run_detection.py
│   ├── run_tracking.py
│   └── outputs/
│       ├── output_tracked.mp4
│       └── frames/
│
├── slam_simulation/            # Для экспериментов с AirSim или другими симами
│   ├── run_slam.py
│   ├── airsim_env_setup.md
│   └── camera_feed.mp4
│
├── jetson_optimization/        # Оптимизация под Jetson
│   ├── convert_to_tensorrt.py
│   ├── test_fp_
