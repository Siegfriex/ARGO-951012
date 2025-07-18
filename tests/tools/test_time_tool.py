from datetime import datetime
from argo.tools.time_tool import get_current_time, TimeToolInput


def test_get_current_time_success():
    """
    Tests the successful case where a valid timezone is provided.
    """
    # 1. 준비 (Arrange)
    test_input = TimeToolInput(timezone="UTC")

    # 2. 실행 (Act)
    result = get_current_time(test_input)

    # 3. 단언 (Assert)
    assert result.error is None
    assert result.timezone == "UTC"
    # 반환된 시간이 유효한 ISO 8601 형식인지 확인
    assert datetime.fromisoformat(result.iso_8601_time.replace("Z", "+00:00"))


def test_get_current_time_default_timezone():
    """
    Tests the default case where no timezone is provided.
    """
    test_input = TimeToolInput()  # 기본값 사용
    result = get_current_time(test_input)
    assert result.error is None
    assert result.timezone == "Asia/Seoul"
    assert datetime.fromisoformat(result.iso_8601_time)


def test_get_current_time_invalid_timezone():
    """
    Tests the failure case where an invalid timezone is provided.
    """
    test_input = TimeToolInput(timezone="Invalid/Timezone")
    result = get_current_time(test_input)
    assert result.error == "Invalid timezone: 'Invalid/Timezone'"
    assert result.iso_8601_time == "N/A"
