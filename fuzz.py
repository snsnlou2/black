
"Property-based tests for Black.\n\nBy Zac Hatfield-Dodds, based on my Hypothesmith tool for source code\ngeneration.  You can run this file with `python`, `pytest`, or (soon)\na coverage-guided fuzzer I'm working on.\n"
import hypothesmith
from hypothesis import HealthCheck, given, settings, strategies as st
import black

@settings(max_examples=1000, derandomize=True, deadline=None, suppress_health_check=HealthCheck.all())
@given(src_contents=(hypothesmith.from_grammar() | hypothesmith.from_node()), mode=st.builds(black.FileMode, line_length=(st.just(88) | st.integers(0, 200)), string_normalization=st.booleans(), is_pyi=st.booleans()))
def test_idempotent_any_syntatically_valid_python(src_contents, mode):
    compile(src_contents, '<string>', 'exec')
    try:
        dst_contents = black.format_str(src_contents, mode=mode)
    except black.InvalidInput:
        return
    black.assert_equivalent(src_contents, dst_contents)
    black.assert_stable(src_contents, dst_contents, mode=mode)
if (__name__ == '__main__'):
    test_idempotent_any_syntatically_valid_python()
