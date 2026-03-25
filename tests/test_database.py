from app.utils.db import save_result, get_all_results

def test ():
    """
    TC09: FRQ9 - Data storage
    """
    save_result("Test message", "Mobile Banking", "positive", 0.98)

    data = get_all_results()
    assert len(data) > 0


def test_no_pii_storage():
    """
    TC15: NFRQ6 - No PII stored
    """
    data = get_all_results()

    for row in data:
        assert "@" not in str(row)  # no emails