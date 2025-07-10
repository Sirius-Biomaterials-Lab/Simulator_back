import logging.config
from contextlib import suppress

import yaml
from starlette_context import context
from starlette_context.errors import ContextDoesNotExistError
from starlette_context.header_keys import HeaderKeys

with open('config/logging.conf.yml', 'r') as f:
    LOGGING_CONFIG = yaml.full_load(f)


class ConsoleFormatter(logging.Formatter):
    def __init__(self):
        # Формат логов согласно требованию
        format_string = "%(asctime)s - [%(levelname)s] -  %(name)s - (%(filename)s).%(funcName)s.%(lineno)d - %(message)s"
        super().__init__(format_string)
    
    def format(self, record: logging.LogRecord) -> str:
        # Добавляем correlation_id в начало сообщения, если он есть в контексте
        with suppress(ContextDoesNotExistError):
            if corr_id := context.get(HeaderKeys.correlation_id, None):
                original_msg = record.getMessage()
                record.msg = f'[{corr_id}] {original_msg}'
                record.args = None

        return super().format(record)


logger = logging.getLogger('backend_logger')
