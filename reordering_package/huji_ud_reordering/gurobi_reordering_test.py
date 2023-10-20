import ReorderingNew as R
import UDLib as U
import json
import os


def run_test(test_name: str, input_file: str, estimates_file: str, expected_file: str):
    print(f'Running test: {test_name}...')
    trees = U.conllu2trees(input_file)

    with open(estimates_file, 'r', encoding='utf-8') as est:
        estimates = json.load(est)

    results = [R.reorder_tree(tree, estimates) for tree in trees]

    with open(expected_file, "r", encoding="utf-8") as file:
        expected_list = file.read().split("\n")

    assert len(results) == len(expected_list)

    for i, (result, expected) in enumerate(zip(results, expected_list)):
        print(f'Subtest {i}:')
        print(f'Expected: {expected}')
        print(f'Result: {result.get_sentence()}')
        assert expected.lower() == result.get_sentence().lower()
        print("Done!")


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    input_1_path = f'{script_dir}/tests/test_1.input'
    expected_1_path = f'{script_dir}/tests/test_1.expected'
    run_test("Simple test", input_1_path, f'{script_dir}/tests/test_1.estimates_1', expected_1_path)
    run_test("Subtree without rules", input_1_path, f'{script_dir}/tests/test_1.estimates_2', expected_1_path)
    run_test("Simple test with thresholds", input_1_path, f'{script_dir}/tests/test_1.estimates_3', expected_1_path)
    run_test("Subtree without rules with thresholds", input_1_path, f'{script_dir}/tests/test_1.estimates_4',
             expected_1_path)
