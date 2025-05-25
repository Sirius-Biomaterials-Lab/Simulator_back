class UserNotFoundException(Exception):
    detail = 'User not found'


class UserNotCorrectPasswordException(Exception):
    detail = 'Password is incorrect'


class TokenExpire(Exception):
    detail = 'Token expired'


class DataNotCorrect(Exception):
    detail = 'Data not correct'


class DataNotFound(Exception):
    detail = 'Data not found'
