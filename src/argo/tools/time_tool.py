from pydantic import BaseModel, Field
from datetime import datetime
import pytz


class TimeToolInput(BaseModel):
    timezone: str = Field(
        default="Asia/Seoul",
        description="The timezone to get the current time for, e.g., 'Asia/Seoul' or 'UTC'.",
    )


class TimeToolOutput(BaseModel):
    iso_8601_time: str = Field(..., description="The current time in ISO 8601 format.")
    timezone: str = Field(
        ..., description="The timezone used for the time calculation."
    )
    error: str | None = Field(
        default=None, description="Error message if the timezone is unknown."
    )


def get_current_time(input: TimeToolInput) -> TimeToolOutput:
    """
    지정된 시간대의 현재 시간을 ISO 8601 형식으로 반환합니다.
    """
    try:
        tz = pytz.timezone(input.timezone)
        now = datetime.now(tz)
        formatted_time = now.isoformat()

        return TimeToolOutput(iso_8601_time=formatted_time, timezone=input.timezone)
    except pytz.UnknownTimeZoneError:
        return TimeToolOutput(
            iso_8601_time="N/A",
            timezone=input.timezone,
            error=f"Invalid timezone: '{input.timezone}'",
        )
