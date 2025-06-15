"""
Custom Pylint plugin to detect uncontrolled randomness in ML code.
"""
from pylint.checkers import BaseChecker

CALLS_TO_CHECK = {
   "train_test_split"
}

class RandomnessUncontrolledMLSmellChecker(BaseChecker):
    """
    Custom Pylint checker to detect uncontrolled randomness in ML code.
    """
    name = "ml-smell-randomness-uncontrolled"
    priority = -1

    msgs = {
        "W9001": (
            "Randomness used in %s without setting random state/seed.",
            "ml-smell-randomness-uncontrolled",
            "Warns if the code contains the randomness uncontrolled ML code smell."
        )
    }

    def visit_call(self, node):
        """
        Check for function calls that should have a random state or seed set.
        """
        if not hasattr(node.func, "attrname") and not hasattr(node.func, "name"):
            return

        func = node.func.attrname if hasattr(node.func, "attrname") else node.func.name
        if func in CALLS_TO_CHECK:
            has_random_state = any(
                (keyword.arg in ("random_state", "seed"))
                and keyword.value is not None
                for keyword in node.keywords
            )

            if not has_random_state:
                self.add_message(
                "ml-smell-randomness-uncontrolled",
                node=node,
                args=(func,)
                )

def register(linter):
    """
    Register the checker with the linter.
    """
    linter.register_checker(RandomnessUncontrolledMLSmellChecker(linter))