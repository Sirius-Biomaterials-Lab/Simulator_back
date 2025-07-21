import sentry_sdk
def start_sentry():
    sentry_sdk.init(
        dsn="https://be26a19afc26dfc6a57aab57834b6246@o4509357555515392.ingest.de.sentry.io/4509655652106320",
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=True,
    )
