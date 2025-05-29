from fastapi import status


class UserNotFoundException(Exception):
    detail = 'User not found'


class UserNotCorrectException(Exception):
    status_code = status.HTTP_401_UNAUTHORIZED,
    detail = "Invalid username or password",
    headers = {"WWW-Authenticate": "Basic"},


class TokenExpire(Exception):
    detail = 'Token expired'


class DataNotCorrect(Exception):
    detail = 'Data not correct'


class DataNotFound(Exception):
    detail = 'Data not found'
