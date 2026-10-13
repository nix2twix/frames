
"""Загрузка, проверка и сериализация конфигураций атак FRAMES."""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, BinaryIO, TextIO


SCHEMA_VERSION = 1

ATTACK_TYPES = (
    "FGSM",
    "PGD",
    "Carlini-Wagner",
    "Geometric Attack",
)

SCENARIO_TYPES = (
    "impersonation",
    "deauth",
)

# Соответствие сценария условию успешности атаки.
SUCCESS_CRITERIA = {
    "impersonation": "target_similarity_crosses_threshold",
    "deauth": "self_similarity_falls_below_threshold",
}

# Выбор цели соответствует текущей реализации приложения:
# при подмене используется следующее изображение в батче,
# при обходе аутентификации — исходное изображение.
TARGET_SELECTIONS = {
    "impersonation": "next_in_batch",
    "deauth": "self",
}

# Параметры и диапазоны согласованы с текущими элементами UI.
PARAMETER_SPECS = {
    "FGSM": {
        "epsilon": {
            "type": float,
            "min": 0.0,
            "max": 0.3,
            "default": 0.08,
        },
    },
    "PGD": {
        "epsilon": {
            "type": float,
            "min": 0.0,
            "max": 0.3,
            "default": 0.08,
        },
        "alpha": {
            "type": float,
            "min": 0.001,
            "max": 0.05,
            "default": 0.01,
        },
        "iters": {
            "type": int,
            "min": 1,
            "max": 100,
            "default": 15,
        },
    },
    "Carlini-Wagner": {
        "epsilon": {
            "type": float,
            "min": 0.0,
            "max": 0.3,
            "default": 0.08,
        },
        "c": {
            "type": float,
            "min": 0.01,
            "max": 100.0,
            "default": 1.0,
        },
        "lr": {
            "type": float,
            "min": 1e-5,
            "max": 1.0,
            "default": 0.01,
        },
        "iters": {
            "type": int,
            "min": 10,
            "max": 1000,
            "default": 100,
        },
    },
    "Geometric Attack": {
        "strength": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
        },
    },
}

DEFAULT_THRESHOLD = 0.5


class ConfigurationError(ValueError):
    """Ошибка структуры или значений конфигурации."""


def default_config(
    attack_type: str = "FGSM",
    scenario_type: str = "impersonation",
) -> dict[str, Any]:
    """Создаёт конфигурацию с параметрами по умолчанию."""

    if attack_type not in ATTACK_TYPES:
        raise ConfigurationError(
            f"Неизвестный тип атаки: {attack_type!r}."
        )

    if scenario_type not in SCENARIO_TYPES:
        raise ConfigurationError(
            f"Неизвестный сценарий: {scenario_type!r}."
        )

    parameters = {
        name: spec["default"]
        for name, spec in PARAMETER_SPECS[attack_type].items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "attack": {
            "type": attack_type,
            "parameters": parameters,
        },
        "scenario": {
            "type": scenario_type,
            "success_criterion": SUCCESS_CRITERIA[scenario_type],
            "threshold": DEFAULT_THRESHOLD,
            "target_selection": TARGET_SELECTIONS[scenario_type],
        },
    }


def _require_object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Поле {name!r} должно быть JSON-объектом."
        )
    return value


def _check_keys(
    obj: dict,
    *,
    required: set[str],
    allowed: set[str],
    name: str,
) -> None:
    missing = required - obj.keys()
    unknown = obj.keys() - allowed

    if missing:
        fields = ", ".join(sorted(missing))
        raise ConfigurationError(
            f"В объекте {name!r} отсутствуют поля: {fields}."
        )

    if unknown:
        fields = ", ".join(sorted(unknown))
        raise ConfigurationError(
            f"В объекте {name!r} неизвестные поля: {fields}."
        )


def _validate_number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
    expected_type: type = float,
) -> int | float:
    # bool является подклассом int в Python, поэтому исключаем его отдельно.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"Параметр {name!r} должен быть числом."
        )

    if not math.isfinite(value):
        raise ConfigurationError(
            f"Параметр {name!r} должен быть конечным числом."
        )

    if expected_type is int and not isinstance(value, int):
        raise ConfigurationError(
            f"Параметр {name!r} должен быть целым числом."
        )

    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"Параметр {name!r} должен находиться "
            f"в диапазоне [{minimum}, {maximum}]. Получено: {value}."
        )

    return int(value) if expected_type is int else float(value)


def validate_config(config: Any) -> dict[str, Any]:
    """
    Проверяет структуру и значения конфигурации.

    Возвращает независимую копию конфигурации с проверенными числами.
    При ошибке выбрасывает ConfigurationError.
    """

    config = _require_object(config, "root")
    _check_keys(
        config,
        required={"schema_version", "attack", "scenario"},
        allowed={"schema_version", "attack", "scenario"},
        name="root",
    )

    version = config["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SCHEMA_VERSION
    ):
        raise ConfigurationError(
            f"Неподдерживаемая версия конфигурации: {version!r}. "
            f"Ожидается версия {SCHEMA_VERSION}."
        )

    attack = _require_object(config["attack"], "attack")
    _check_keys(
        attack,
        required={"type", "parameters"},
        allowed={"type", "parameters"},
        name="attack",
    )

    attack_type = attack["type"]
    if not isinstance(attack_type, str) or attack_type not in ATTACK_TYPES:
        raise ConfigurationError(
            f"Неизвестный тип атаки: {attack_type!r}. "
            f"Допустимые типы: {', '.join(ATTACK_TYPES)}."
        )

    parameters = _require_object(
        attack["parameters"],
        "attack.parameters",
    )
    specs = PARAMETER_SPECS[attack_type]
    _check_keys(
        parameters,
        required=set(specs),
        allowed=set(specs),
        name="attack.parameters",
    )

    checked_parameters = {}
    for name, spec in specs.items():
        checked_parameters[name] = _validate_number(
            parameters[name],
            name=name,
            minimum=spec["min"],
            maximum=spec["max"],
            expected_type=spec["type"],
        )

    scenario = _require_object(config["scenario"], "scenario")
    _check_keys(
        scenario,
        required={
            "type",
            "success_criterion",
            "threshold",
            "target_selection",
        },
        allowed={
            "type",
            "success_criterion",
            "threshold",
            "target_selection",
        },
        name="scenario",
    )

    scenario_type = scenario["type"]
    if (
        not isinstance(scenario_type, str)
        or scenario_type not in SCENARIO_TYPES
    ):
        raise ConfigurationError(
            f"Неизвестный сценарий: {scenario_type!r}. "
            f"Допустимые сценарии: {', '.join(SCENARIO_TYPES)}."
        )

    expected_criterion = SUCCESS_CRITERIA[scenario_type]
    if scenario["success_criterion"] != expected_criterion:
        raise ConfigurationError(
            f"Для сценария {scenario_type!r} условие успешности "
            f"должно быть {expected_criterion!r}."
        )

    expected_target = TARGET_SELECTIONS[scenario_type]
    if scenario["target_selection"] != expected_target:
        raise ConfigurationError(
            f"Для сценария {scenario_type!r} способ выбора цели "
            f"должен быть {expected_target!r}."
        )

    threshold = _validate_number(
        scenario["threshold"],
        name="threshold",
        minimum=0.0,
        maximum=1.0,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "attack": {
            "type": attack_type,
            "parameters": checked_parameters,
        },
        "scenario": {
            "type": scenario_type,
            "success_criterion": expected_criterion,
            "threshold": threshold,
            "target_selection": expected_target,
        },
    }


def create_config(
    attack_type: str,
    parameters: dict[str, Any],
    scenario_type: str,
    threshold: float,
) -> dict[str, Any]:
    """Создаёт и проверяет конфигурацию из текущих настроек UI."""

    config = {
        "schema_version": SCHEMA_VERSION,
        "attack": {
            "type": attack_type,
            "parameters": deepcopy(parameters),
        },
        "scenario": {
            "type": scenario_type,
            "success_criterion": SUCCESS_CRITERIA.get(scenario_type),
            "threshold": threshold,
            "target_selection": TARGET_SELECTIONS.get(scenario_type),
        },
    }
    return validate_config(config)


def parse_config(json_text: str | bytes) -> dict[str, Any]:
    """Разбирает JSON-текст и проверяет конфигурацию."""

    if isinstance(json_text, bytes):
        try:
            json_text = json_text.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ConfigurationError(
                "Файл конфигурации должен быть в кодировке UTF-8."
            ) from exc

    if not isinstance(json_text, str):
        raise ConfigurationError(
            "Содержимое конфигурации должно быть текстом или байтами."
        )

    try:
        raw_config = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Некорректный JSON: строка {exc.lineno}, "
            f"столбец {exc.colno}: {exc.msg}."
        ) from exc

    return validate_config(raw_config)


def load_config(source: Any) -> dict[str, Any]:
    """
    Загружает конфигурацию из пути, JSON-строки, байтов
    или объекта Streamlit UploadedFile.
    """

    if isinstance(source, bytes):
        return parse_config(source)

    if isinstance(source, Path):
        try:
            return parse_config(source.read_bytes())
        except OSError as exc:
            raise ConfigurationError(
                f"Не удалось прочитать файл конфигурации: {exc}."
            ) from exc

    # Streamlit UploadedFile предоставляет getvalue().
    if hasattr(source, "getvalue"):
        return parse_config(source.getvalue())

    if hasattr(source, "read"):
        return parse_config(source.read())

    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return parse_config(source)

        try:
            return parse_config(Path(source).read_bytes())
        except OSError as exc:
            raise ConfigurationError(
                f"Не удалось прочитать файл конфигурации: {exc}."
            ) from exc

    raise ConfigurationError(
        "Источник конфигурации должен быть путём, JSON-текстом, "
        "байтами или загруженным файлом."
    )


def serialize_config(config: dict[str, Any]) -> str:
    """Проверяет конфигурацию и преобразует её в JSON-текст."""

    checked_config = validate_config(config)
    return json.dumps(
        checked_config,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def save_config(
    config: dict[str, Any],
    destination: str | Path,
) -> None:
    """Проверяет конфигурацию и сохраняет её в UTF-8 JSON-файл."""

    json_text = serialize_config(config)
    path = Path(destination)

    try:
        path.write_text(json_text + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Не удалось сохранить конфигурацию: {exc}."
        ) from exc
