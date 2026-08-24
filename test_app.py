from app import predict_result
def test_student_pass():
    assert predict_result(75) == "PASS"
def test_student_fail():
    assert predict_result(35) == "FAIL"