topics = {
    "inbound": {  # Входящие команды (управление дроном)
        "actuator_motors": "/fmu/in/actuator_motors",  # Управление моторами
        "actuator_servos": "/fmu/in/actuator_servos",  # Управление сервоприводами
        "arming_check_reply": "/fmu/in/arming_check_reply",  # Ответ на команду проверки готовности к взлету
        "aux_global_position": "/fmu/in/aux_global_position",  # Глобальная позиция вспомогательных устройств
        "config_control_setpoints": "/fmu/in/config_control_setpoints",  # Настройка контрольных точек управления
        "config_overrides_request": "/fmu/in/config_overrides_request",  # Запрос на изменение параметров управления
        "distance_sensor": "/fmu/in/distance_sensor",  # Данные дальномера
        "goto_setpoint": "/fmu/in/goto_setpoint",  # Команда перемещения в заданную точку
        "manual_control_input": "/fmu/in/manual_control_input",  # Входные данные для ручного управления
        "message_format_request": "/fmu/in/message_format_request",  # Запрос формата сообщений
        "mode_completed": "/fmu/in/mode_completed",  # Подтверждение завершения режима работы
        "obstacle_distance": "/fmu/in/obstacle_distance",  # Данные о препятствиях
        "offboard_control_mode": "/fmu/in/offboard_control_mode",  # Команды режима offboard
        "onboard_computer_status": "/fmu/in/onboard_computer_status",  # Статус бортового компьютера
        "register_ext_component_request": "/fmu/in/register_ext_component_request",  # Запрос на регистрацию внешнего компонента
        "sensor_optical_flow": "/fmu/in/sensor_optical_flow",  # Данные оптического потока
        "telemetry_status": "/fmu/in/telemetry_status",  # Статус телеметрии
        "trajectory_setpoint": "/fmu/in/trajectory_setpoint",  # Задание траектории
        "unregister_ext_component": "/fmu/in/unregister_ext_component",  # Отмена регистрации внешнего компонента
        "vehicle_attitude_setpoint": "/fmu/in/vehicle_attitude_setpoint",  # Задание ориентации аппарата
        "vehicle_command": "/fmu/in/vehicle_command",  # Команды управления полетом
        "vehicle_command_mode_executor": "/fmu/in/vehicle_command_mode_executor",  # Контроль выполнения команд
        "vehicle_mocap_odometry": "/fmu/in/vehicle_mocap_odometry",  # Данные о внешнем слежении за движением
        "vehicle_rates_setpoint": "/fmu/in/vehicle_rates_setpoint",  # Задание угловых скоростей
        "vehicle_thrust_setpoint": "/fmu/in/vehicle_thrust_setpoint",  # Установка тяги
        "vehicle_torque_setpoint": "/fmu/in/vehicle_torque_setpoint",  # Установка момента сил
        "vehicle_visual_odometry": "/fmu/in/vehicle_visual_odometry",  # Данные визуальной одометрии
    },
    "outbound": {  # Исходящие данные (телеметрия)
        "battery_status": "/fmu/out/battery_status",  # Статус заряда аккумулятора
        "estimator_status_flags": "/fmu/out/estimator_status_flags",  # Флаги статуса оценивания положения
        "failsafe_flags": "/fmu/out/failsafe_flags",  # Флаги аварийного режима
        "manual_control_setpoint": "/fmu/out/manual_control_setpoint",  # Установленные значения ручного управления
        "position_setpoint_triplet": "/fmu/out/position_setpoint_triplet",  # Параметры заданных точек маршрута
        "sensor_combined": "/fmu/out/sensor_combined",  # Данные с сенсоров
        "timesync_status": "/fmu/out/timesync_status",  # Статус синхронизации времени
        "vehicle_attitude": "/fmu/out/vehicle_attitude",  # Ориентация дрона
        "vehicle_command_ack": "/fmu/out/vehicle_command_ack",  # Подтверждение выполнения команды
        "vehicle_control_mode": "/fmu/out/vehicle_control_mode",  # Режим управления дроном
        "vehicle_global_position": "/fmu/out/vehicle_global_position",  # Глобальная позиция
        "vehicle_gps_position": "/fmu/out/vehicle_gps_position",  # Данные GPS
        "vehicle_land_detected": "/fmu/out/vehicle_land_detected",  # Обнаружение посадки
        "vehicle_local_position": "/fmu/out/vehicle_local_position",  # Локальная позиция дрона
        "vehicle_odometry": "/fmu/out/vehicle_odometry",  # Одометрия движения
        "vehicle_status_v1": "/fmu/out/vehicle_status_v1",  # Общий статус дрона
    },
    "model": {  # Топики, связанные с моделью дрона
        "x500_odometry": "/model/x500/odometry",  # Одометрия модели X500
    },
    "system": {  # Системные топики
        "parameter_events": "/parameter_events",  # События изменения параметров
        "rosout": "/rosout",  # Логи ROS2
    },
}