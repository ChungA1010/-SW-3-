class AppError(Exception):
    def __init__(self, user_message: str, status_code: int = 400) -> None:
        self.user_message = user_message
        self.status_code = status_code
        super().__init__(user_message)

