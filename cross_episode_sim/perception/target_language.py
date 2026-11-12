"""Public visual vocabulary for abstract household task categories.

These descriptions are task conventions, not simulator object bindings. Keep
semantic labels for scoring and tracking; expand them only at model inputs.
"""
import re


GROCERY_DESCRIPTIONS = {
    'packaged drink': 'drink bottles, canned beverages, or clearly labeled beverage cartons, excluding plain rectangular snack boxes',
    'packaged snack': (
        'small portable rectangular blocks or boxes representing snacks '
        '(they may be plain solid-colored blocks without labels or food graphics; '
        'exclude baskets, furniture, doors, bottles and cups)'
    ),
}


def visual_target_text(text):
    """Expand singular/plural category mentions, preserving location qualifiers."""
    for category, description in GROCERY_DESCRIPTIONS.items():
        # Keep the already-qualified drink call sites unchanged; navigation's
        # GroceryPointer still uses its existing bottle/carton description.
        if category != 'packaged snack':
            continue
        text = re.sub(r'\b' + re.escape(category) + r's?\b',
                      lambda match: description, text, flags=re.IGNORECASE)
    return text
