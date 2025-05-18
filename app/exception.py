class UserNotFoundException(Exception):
    detail = 'User not found'


class UserNotCorrectPasswordException(Exception):
    detail = 'Password is incorrect'


class TokenExpire(Exception):
    detail = 'Token expired'


class TokenNotCorrect(Exception):
    detail = 'Token not correct or expired'


class DataNotFound(Exception):
    detail = 'Data not found'
