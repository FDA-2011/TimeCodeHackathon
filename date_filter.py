import logging
from datetime import timedelta
import zoneinfo
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__) # логи
MSK = zoneinfo.ZoneInfo("Europe/Moscow") # время по МСК


def check_period(pub_date_str, period_start, period_end):
    if not pub_date_str or not str(pub_date_str).strip():
        return False, "дата_отсутствует" # если нет даты

    try:
        dt = dateutil_parser.parse(str(pub_date_str)) # парсим дату

        if dt.tzinfo is None: # если нет часового пояса
            dt = dt.replace(tzinfo=MSK)
        else:
            dt = dt.astimezone(MSK) # иначе ставим МСК

        if dt <= period_start: # если раньше начала периода откидываем
            return False, f"вне_периода_раньше|{dt.isoformat()}"
        if dt > period_end: # если позже тоже выбрасываем
            return False, f"вне_периода_позже|{dt.isoformat()}"

        return True, f"в_периоде|{dt.isoformat()}" # если все ок возвращаем это

    except Exception as e:
        logger.warning(f"Не удалось распарсить дату '{pub_date_str}': {e}") # # запоминаем ошибку
        return False, f"ошибка_парсинга_даты|{str(e)[:100]}"


def build_24h_filter(control_time_msk):
    if control_time_msk.tzinfo is None:
        raise ValueError("control_time_msk должен содержать часовой пояс (МСК)") # проверяем пояс

    period_start = control_time_msk - timedelta(hours=24) # дата старта периода

    def check(pub_date_str):
        return check_period(pub_date_str, period_start, control_time_msk) # запускаем чекер

    return check # возвращаем обновленный чекер
