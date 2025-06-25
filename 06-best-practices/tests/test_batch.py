import batch

def test_read_data_main():
    actual_output = batch.read_data_main(2023, 3)
    expected_output = 14.203865642696089
    assert actual_output == expected_output