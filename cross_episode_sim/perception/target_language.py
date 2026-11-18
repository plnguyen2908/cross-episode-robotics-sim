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


# Storage cabinets are the lower (base) cabinets the arm can reach. Wall-mounted
# upper cabinets are not storage targets (live fix56: pointing chose a wall
# cabinet handle 1.58 m up). Model inputs only; labels and scoring unchanged.
# Prefer the biggest one (live fix57 opened a short compartment under the
# stove top; a sideways bowl barely fits through its opening).
STORAGE_CABINET = ('kitchen storage cabinet (a lower base cabinet below the countertop, '
                   'not a wall-mounted upper cabinet; prefer the largest, tallest '
                   'cabinet door, not a small compartment under the stove or a drawer)')


def visual_target_text(text):
    """Expand singular/plural category mentions, preserving location qualifiers."""
    if 'kitchen storage cabinet' in text and 'wall-mounted' not in text:
        text = re.sub(r'\bkitchen storage cabinets?\b', STORAGE_CABINET, text, count=1)
    for category, description in GROCERY_DESCRIPTIONS.items():
        # Keep the already-qualified drink call sites unchanged; navigation's
        # GroceryPointer still uses its existing bottle/carton description.
        if category != 'packaged snack':
            continue
        text = re.sub(r'\b' + re.escape(category) + r's?\b',
                      lambda match: description, text, flags=re.IGNORECASE)
    return text
