"""Query-text constraint parsing and rule-based plan verification.

Pure functions (no agent state) that:
- detect which (day, slot) the traveler explicitly requested,
- parse per-slot numeric/categorical constraints from the query text, and
- verify selected options against clearly attributable constraints.

Complex and cross-traveler conditions are left to the caller's LLM verification.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .matching import normalize_city, to_float

_ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10,
}
_CARDINAL_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_DAY_VALUE_PAT = rf"(?:\d+(?:st|nd|rd|th)?|{'|'.join((*_ORDINAL_MAP, *_CARDINAL_MAP))})"
_DAY_LIST_PAT = rf"{_DAY_VALUE_PAT}(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*|[–—-]\s*|to\s+)(?:days?\s*)?{_DAY_VALUE_PAT})*"
_DAY_PATTERN = re.compile(
    rf"\b(?:days?\s*[-:]?\s*(?P<after>{_DAY_LIST_PAT})|(?P<before>{_DAY_VALUE_PAT})[- ]+days?)\b"
)
_SLOT_NAMES = ("breakfast", "lunch", "dinner", "accommodation")
# Synonyms that map to canonical slot names (for extraction only).
_SLOT_SYNONYMS = {
    "stay": "accommodation", "lodge": "accommodation", "hotel": "accommodation",
    "private room": "accommodation", "shared room": "accommodation",
    "entire home/apt": "accommodation", "entire home": "accommodation",
    "entire apartment": "accommodation", "entire house": "accommodation",
}
_SLOT_PATTERN = '|'.join(
    re.escape(slot).replace(r'\ ', r'\s+')
    for slot in sorted((*_SLOT_NAMES, *_SLOT_SYNONYMS), key=len, reverse=True)
)
_KNOWN_CUISINES = (
    'Fast Food', 'American', 'Bakery', 'BBQ', 'Cafe', 'Chinese',
    'Desserts', 'French', 'Indian', 'Italian', 'Mediterranean',
    'Mexican', 'Pizza', 'Seafood', 'Tea',
)
_CUISINE_NAME_PATTERN = '(?:' + '|'.join(re.escape(name.lower()) for name in _KNOWN_CUISINES) + ')'
_CUISINE_LIST_PATTERN = rf'{_CUISINE_NAME_PATTERN}(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*){_CUISINE_NAME_PATTERN})*'
_NO_OTHER_CUISINES = r'\band\s+no\s+(?:other|additional)\s+cuisine(?:s|\s+types?)\b'
_TARGET_REQUEST_PATTERN = (
    r"(?:i|we)(?:'d|\s+would)?\s+(?:like|want|need|prefer|hope)\b"
    r"|(?:i(?:'m|\s+am)|we(?:'re|\s+are))\s+(?:looking|seeking)\b"
)
_DISALLOW = r"(?:(?:doesn't|does not|don't|do not|not)\s+(?:allow(?:ing)?|accept(?:ing)?)|prohibits?|disallows?)\s+(?:any\s+)?"
_CHILDREN_UNDER_10 = (
    r'(?:children|kids)(?:\s+under[-\s]*(?:(?:the\s+)?age\s+of\s+)?(?:10|ten)\b'
    r'|(?!\s+(?:(?:under|over|below|above|younger|older|aged?|between)(?=[\s\d-]|$)|of\s+(?:the\s+)?age\b|[<>=≤≥]|\d)))'
)
_HOUSE_RULE_PATTERNS = {
    'no_visitors': rf'\b(?:{_DISALLOW}(?:extra\s+)?visitors|no[- ]?(?:extra\s+)?visitors|non[- ]visitor[- ]friendly)\b',
    'no_smoking': rf'\b(?:{_DISALLOW}smoking|non[- ]?smoking|no[- ]?smoking)\b',
    'no_parties': rf'\b(?:{_DISALLOW}part(?:ies|y)|(?:no|non)[- ]?part(?:ies|y))\b',
    'no_pets': rf'\b(?:{_DISALLOW}pets|no[- ]?pets|pet[- ]free)\b',
    'no_children_under_10': rf'\b(?:{_DISALLOW}{_CHILDREN_UNDER_10}|no[- ]?{_CHILDREN_UNDER_10})\b',
}


def _analyze_joining_query_slots(
    query_text: str,
    plan: list[dict[str, Any]],
    base_plan: list[dict[str, Any]] | None = None,
    base_traveler_name: str | None = None,
    *,
    prior_traveler_queries: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str], list[tuple[int, str]]]:
    """Return filled items, single-line TODO requirements, and requested keys.

    Only an explicit, unchanged same-day/same-item base arrangement can fill an
    unselected target. This does not copy the base plan into the current plan.
    """
    filled: list[str] = []
    missing: list[str] = []
    requests = _parse_query_slot_constraints(
        query_text, base_traveler_name, prior_traveler_queries=prior_traveler_queries,
    )
    selections = _build_selection_map(plan)
    base_selections = _build_selection_map(base_plan or [])
    days = {entry['days']: entry for entry in plan}
    base_days = {entry['days']: entry for entry in base_plan or []}
    for (day, slot), constraints in requests.items():
        if (day, slot) in selections:
            option = selections[(day, slot)]
            value = days[day].get(slot) or option.get('name', '-')
            filled.append(f"Day {day} {slot}: {value}")
            continue
        if constraints['retain_base']:
            option = base_selections.get((day, slot), {})
            value = base_days.get(day, {}).get(slot)
            # Supplied base plans may contain only "<name>, <city>" strings.
            supplied_value = (isinstance(value, str) and ',' in value
                              and all(part.strip() not in ('', '-') for part in value.rsplit(',', 1)))
            selected_name = option.get('name')
            if supplied_value or (isinstance(selected_name, str) and selected_name.strip() not in ('', '-')):
                filled.append(f"Day {day} {slot}: {value if supplied_value else selected_name} (retained base arrangement)")
                continue
        missing.append(f"Day {day} {slot}: {constraints['requirement_text']}")
    return filled, missing, list(requests)


def _days_in_mention(match: re.Match[str]) -> list[int]:
    """Read explicit trip-day spellings, lists and inclusive day ranges."""
    text = match.group('after') or match.group('before')
    values = list(re.finditer(_DAY_VALUE_PAT, text))
    days: list[int] = []
    for index, value in enumerate(values):
        token = value.group()
        day = (int(re.match(r'\d+', token).group()) if token[0].isdigit()
               else {**_ORDINAL_MAP, **_CARDINAL_MAP}[token])
        if index and re.fullmatch(r'\s*(?:[–—-]|to)\s*', text[values[index - 1].end():value.start()]):
            if days and day > days[-1]:
                days.extend(range(days[-1] + 1, day + 1))
                continue
        if day > 0 and day not in days:
            days.append(day)
    return days


def _normalized_query_text(text: str) -> str:
    """Normalize for parsing while keeping offsets into the original text."""
    normalized = text.lower().replace('’', "'")
    # "The second day's dinner" owns a date, not another traveler's dinner.
    return re.sub(
        rf"(?P<date>{_DAY_PATTERN.pattern})'s\b",
        lambda match: match.group('date') + '  ', normalized,
    )


def _reference_context(text: str) -> bool:
    """Recognize a comparison operand, not a later coordinated request."""
    # Only a clear new request resets reference ownership. A plain "and" also
    # joins reference lists, so it cannot by itself introduce another target.
    resets = list(re.finditer(
        rf'(?:,\s*(?:and\s+)?|\b(?:and|then)\s+)(?:please\s+)?(?:choose|select|find|book|keep|retain|reuse)\b'
        rf'|,\s*(?:and\s+)?(?:(?:on|for)\s+)?(?:the\s+)?{_DAY_PATTERN.pattern}', text,
    ))
    if resets:
        text = text[resets[-1].end():]
    reference_item = (
        rf'(?:days?\s*[-:]?\s*{_DAY_VALUE_PAT}\b|{_DAY_VALUE_PAT}[- ]+days?\b|(?:{_SLOT_PATTERN})\b'
        r'|(?:restaurant|room|price|cost|rate|rating|cuisines?|house|rules?)\b|(?:no|non)[- ](?:smoking|pets|parties|visitors|children|kids)\b)'
    )
    return bool(re.search(
        r"\b(?:than|compared|relative\s+to)\b"
        r"|\b(?:same|similar|different)\b[^,;!?]*\b(?:as|from|to)\b"
        r"|\b[a-z][\w'-]*\s+(?:chose|selected|booked|had|used)\b"
        r"|\b(?:chosen|selected|booked)\s+by\b"
        rf"|\b[\w'-]+'s\s*(?={reference_item}|(?:on|for|from)\b|$)"
        rf"|\b(?:his|her|their)\s*(?={reference_item}|$)",
        text,
    ))


def _day_slot_groups(paragraph: str) -> list[tuple[int, int, list[tuple[int, str]]]]:
    """Locate target groups while retaining offsets into the original text."""
    p = _normalized_query_text(paragraph)
    results: list[tuple[int, int, list[tuple[int, str]]]] = []
    # Only tightly coordinated slots share a suffix; prose starts a new group.
    list_item = rf'(?:[ \t]*,[ \t]*(?:and[ \t]+)?|[ \t]+(?:and|plus|&)[ \t]+)\b(?:{_SLOT_PATTERN})\b'
    possessive = re.compile(r"'s\s*$")
    for clause_start, clause_end in _constraint_clause_spans(p):
        clause = p[clause_start:clause_end]
        dates = list(_DAY_PATTERN.finditer(clause))
        groups = list(re.finditer(rf'\b(?:{_SLOT_PATTERN})\b(?:{list_item})*', clause))
        suffix_dates: set[int] = set()
        for index, group in enumerate(groups):
            slots = re.findall(rf'\b(?:{_SLOT_PATTERN})\b', group.group())
            prefix = next((date for date in reversed(dates) if date.end() <= group.start()), None)
            next_slot = groups[index + 1].start() if index + 1 < len(groups) else len(clause)
            suffix = next((date for date in dates if group.end() <= date.start() < next_slot), None)
            if suffix:
                suffix_text = clause[group.end():suffix.start()]
                same_place = re.fullmatch(
                    r"\s+(?:at|in)\s+(?:the\s+)?same\s+place\s+as\s+(?P<person>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)*)\s+(?:on|for)\s+(?:the\s+)?",
                    suffix_text,
                )
                # With no earlier target date, "stay at the same place as
                # Rowan on day 5" dates the requested stay. A possessed item
                # or another explicitly dated target is not this shorthand.
                dates_same_stay = (
                    prefix is None and same_place is not None
                    and all(_SLOT_SYNONYMS.get(slot, slot) == 'accommodation' for slot in slots)
                    and not _reference_context(same_place.group('person'))
                )
                reference_date = (
                    possessive.search(clause[:suffix.start()]) or _reference_context(suffix_text)
                    or (prefix and re.search(r'\bsame\s*$', clause[:group.start()])
                        and re.match(r'\s+as\b', suffix_text))
                )
                if reference_date and not dates_same_stay:
                    suffix = None
            # "Keep the base plan's day 1 dinner" is itself a request. Later
            # possessed items in a request remain references (including after
            # "use", as in "Day 1 dinner: use Rowan's day 2 lunch restaurant").
            keep_request = index == 0 and re.match(
                r'\s*(?:please\s+)?(?:keep|retain|reuse|use|leave)\b', clause,
            )
            is_reference = (possessive.search(clause[:group.start()])
                            or (prefix and possessive.search(clause[:prefix.start()]))
                            or _reference_context(clause[:group.start()]))
            if is_reference and not keep_request:
                if suffix:
                    suffix_dates.add(suffix.start())
                if prefix and (index == 0 or prefix.start() >= groups[index - 1].end()):
                    suffix_dates.add(prefix.start())
                continue
            if prefix and (prefix.start() in suffix_dates
                           or (possessive.search(clause[:prefix.start()]) and not keep_request)):
                prefix = None
            # An earlier background date ("arrive on day 1, ...") does not take
            # the first item from an explicitly suffix-dated list.
            local_prefix = prefix and (
                not clause[prefix.end():group.start()].strip()
                or re.search(r'(?:^|,|\band\b)\s*(?:(?:on|for)\s+)?(?:the\s+)?$', clause[:prefix.start()])
            )
            # A date introducing the next request is not this list's suffix.
            if suffix and (
                re.search(r'\b(?:and|then)\s+(?:(?:on|for)\s+)?(?:the\s+)?$', clause[group.end():suffix.start()])
                or (index + 1 < len(groups)
                    and re.search(r',\s*(?:(?:on|for)\s+)?(?:the\s+)?$', clause[group.end():suffix.start()])
                    and re.match(r'\s*,?\s*(?:(?:choose|select|find|book)\b|$)', clause[suffix.end():next_slot]))
            ):
                suffix = None
            assignments = [prefix] * len(slots)
            if suffix:
                assignments = [suffix] * len(slots)
                if local_prefix:
                    # "Day 1 lunch and dinner on day 2" keeps the explicit
                    # prefix for earlier items; only the last has a suffix.
                    assignments[:-1] = [prefix] * (len(slots) - 1)
                suffix_dates.add(suffix.start())
            keys: list[tuple[int, str]] = []
            start, end = group.start(), group.end()
            for slot, date in zip(slots, assignments):
                if date is None:
                    continue
                slot = re.sub(r'\s+', ' ', slot)
                if date.start() < start and (index == 0 or date.start() >= groups[index - 1].end()):
                    start = date.start()
                end = max(end, date.end())
                for day in _days_in_mention(date):
                    key = (day, _SLOT_SYNONYMS.get(slot, slot))
                    if key not in keys:
                        keys.append(key)
            if keys:
                results.append((clause_start + start, clause_start + end, keys))
    return results


def _extract_day_slot(paragraph: str) -> list[tuple[int, str]]:
    """Associate explicit target dates with local slots, excluding references."""
    return list(dict.fromkeys(key for _, _, keys in _day_slot_groups(paragraph) for key in keys))


def _normalized_attribute_text(text: str) -> str:
    """Expand attribute contractions without changing target/source parsing."""
    return re.sub(r"\bthat's\b", 'that is', _normalized_query_text(text))


def _has_independent_subject(text: str) -> bool:
    """Recognize a separate statement, not an attached option attribute."""
    p = re.sub(r'^\s*(?:and|so)\s+', '', _normalized_attribute_text(text))
    p = re.sub(r'^\s*also\s+', '', p)
    p = re.split(r'\band\b|[,;!?]|\.(?=\s|$)', p, maxsplit=1)[0].strip()
    if (re.match(_TARGET_REQUEST_PATTERN, p) or re.match(
        r'(?:which|whose|that|is|are|was|were|has|have|costs|serves|offers|choose|select|find|book'
        r'|(?:with\s+)?(?:(?:the|its|a)\s+)?(?:prices?|costs?|rating|cuisines?|room\s+type|maximum\s+occupancy|rules))\b', p,
    ) or re.search(r'\b(?:required|mandatory)\b', p)):
        return False
    # A causal clause has its own subject even when that subject contains a
    # relative clause before the outer predicate. Otherwise a verb inside an
    # attached relative alone does not establish an independent statement.
    return bool(re.search(
        r'^(?:\b(?:since|because)\b.+|(?:(?!\b(?:that|which|whose)\b).)+)'
        r'\b(?:is|are|was|were|has|have|costs|serves|offers)\b', p,
    ))


def _has_ambiguous_constraints(text: str) -> bool:
    """Leave negation, alternatives and references to the full-query LLM check."""
    p = _normalized_attribute_text(text)
    # These negative phrases name affirmative house-rule requirements.
    unqualified = p.replace('home or apartment', 'home').replace('home/apt', 'home')
    # This negation is an explicit set boundary, not an alternative cuisine.
    unqualified = re.sub(_NO_OTHER_CUISINES, '', unqualified)
    for pattern in _HOUSE_RULE_PATTERNS.values():
        unqualified = re.sub(rf'\bsame\s+{pattern}\s+rules?\b', '', unqualified)
        unqualified = re.sub(pattern, '', unqualified)
    # This is also a literal rule request, but supplies no age threshold.
    unqualified = re.sub(rf'\b(?:{_DISALLOW}|no[- ]?)young\s+(?:children|kids)\b', '', unqualified)
    introductions = list(re.finditer(r'(?:,\s*(?:and\s+)?|\band\s+)(?:for|on)\b', unqualified))
    groups = _day_slot_groups(unqualified) if introductions else []
    # A date and its meal can introduce the same target twice. Only an
    # introduction outside one target's source span signals separate scope.
    separate_scope = any(
        not re.fullmatch(r'\s*finally\s*', unqualified[:intro.start()])
        and not any(start <= intro.start() and intro.end() <= end and len(keys) == 1
                    for start, end, keys in groups)
        for intro in introductions
    )
    return bool(re.search(
        r"\b(?:not|no|never|cannot|unable|without|neither|nor|either|or|avoid(?:s|ed|ing)?|exclud(?:e[sd]?|ing)|except|alternative(?:ly|s)?|while|whereas|although|but|unless)\b"
        r"|n't\b|/|\b(?:rather than|instead of|other than|as opposed to|(?:one|any)\s+of|choice\s+of)\b",
        unqualified,
    ) or separate_scope or re.search(r"\w+'s\b|\b(?:join|same|different|similar|than|compared)\b", unqualified)
        or _reference_context(unqualified)
        or any(_has_independent_subject(unqualified[m.end():])
               for m in re.finditer(r'\band\s+', unqualified)))


def _parse_paragraph_constraints(text: str, slot_type: str | None = None) -> dict[str, Any]:
    """Extract clear attributes without turning comparisons into numeric bounds."""
    p = _normalized_attribute_text(text)
    c: dict[str, Any] = {}
    if re.search(r'\bjoin(?:ing)?\b', p):
        c['is_join'] = True
    if re.search(r'\b(?:within|at least|more than|less than|higher|lower|same|different)\b', p):
        c['has_relative_constraint'] = True

    def merge(attributes: dict[str, Any]) -> None:
        # Separate positive predicates can require multiple cuisines.
        if c.get('cuisines_exact') and not attributes.get('cuisines_exact'):
            attributes.pop('cuisines', None)
        elif c.get('cuisines') and attributes.get('cuisines') and not attributes.get('cuisines_exact'):
            names = set(c['cuisines']) | set(attributes['cuisines'])
            attributes['cuisines'] = [name for name in _KNOWN_CUISINES if name in names]
        c.update(attributes)

    joined_place = re.search(
        rf"\bjoin\s+(?:[\w'-]+(?:\s+[\w'-]+)?\s+at\s+)?(?:a|an|the)\s+"
        rf'(?P<place>(?:{_CUISINE_LIST_PATTERN}\s+)?(?:place|spot|restaurant))\b', p,
    )
    if joined_place and not _has_ambiguous_constraints(p[:joined_place.start()]):
        # The explicitly described venue after joining is the requested
        # option, not a reference item. New conditions still require selection.
        merge(_parse_paragraph_constraints(p[:joined_place.start()], slot_type))
        merge(_parse_paragraph_constraints(p[joined_place.start('place'):], slot_type))
        return c

    if _has_ambiguous_constraints(p):
        # Clear attributes before a reference still belong to the target. Do
        # not scan through the reference for its cuisines, prices or ratings.
        reference = re.search(
            r"\b(?:than|compared|relative\s+to|same|different|similar)\b|\b[\w'-]+'s\b"
            r"|\b[a-z][\w'-]*\s+(?:chose|selected|booked|had|used)\b|\b(?:his|her|their)\b", p,
        )
        if reference:
            prefix = p[:reference.start()]
            prefix_clear = not _has_ambiguous_constraints(prefix)

            def owns_continuation(position: int) -> bool:
                context = re.sub(r'\bthat\s+(?:day|night)\b', '', p[reference.start():position])
                # A requested different room type can itself own "that costs";
                # this is not a relative clause attached to a named reference.
                context = re.sub(
                    r'^(?:same|different)\s+room\s+(?:type|category)\s+that\s+(?=costs\b)', '', context,
                )
                return (prefix_clear
                        and not re.search(r"\b(?:that|which|whose|or|not|neither|either|instead)\b|n't\b", context)
                        and not any(_has_independent_subject(part) for part in re.split(r',|\band\b', context)))

            if prefix.strip() and prefix_clear:
                merge(_parse_paragraph_constraints(prefix, slot_type))
            # Literal shared rules can be named alongside a room type or
            # through a dated owner. Require a target relation or a positive
            # target continuation, rather than gathering reference attributes.
            owner = rf"[\w'-]+(?:\s+[\w'-]+)*'s\s+(?:(?:the\s+)?{_DAY_PATTERN.pattern}\s+)?"
            room_and = r'room\s+(?:type|category)\s*(?:,\s*(?:and\s+)?|and\s+|with\s+)(?:the\s+)?(?:same\s+)?'
            for rule, pattern in _HOUSE_RULE_PATTERNS.items():
                relations = (
                    rf'\b(?:shar(?:e|es|ing)|has|have|follows?|enforces?)\s+(?:{owner})?(?:the\s+)?(?:same\s+)?{pattern}\s+rules?\b',
                    rf'\b(?:same\s+|(?:same|different)\s+{room_and}){pattern}\s+rules?\b',
                    rf'\b(?:match(?:es|ing)?|shar(?:e|es|ing))\s+{owner}{room_and}{pattern}\s+rules?\b',
                )
                for relation in relations:
                    for shared in re.finditer(relation, p):
                        attachment = re.search(
                            r'(?:,\s*(?:(?:and|but)\s+)?|\b(?:and|but)\s+|\band\s+that\s+|\bthat\s+also\s+)'
                            r'(?:also\s+)?(?:(?:with|has|have|follows?|enforces?)\s+)?(?:(?:a|the)\s+)?$', p[:shared.start()],
                        )
                        if ((shared.start() <= reference.start()
                             and not _has_ambiguous_constraints(p[:shared.start()]))
                                or (attachment and owns_continuation(attachment.start()))):
                            c[rule] = True
            # Explicit target subjects and clearly coordinated positive
            # predicates can resume after comparisons. Reference-relative
            # clauses and independent subjects cannot donate their attributes.
            subject = ('accommodation|hotel|room' if slot_type == 'accommodation'
                       else 'restaurant' if slot_type in _SLOT_NAMES else '(?!)')
            for attached in re.finditer(
                rf'(?:,\s*(?:(?:and|but)\s+)?|\b(?:and|but)\s+)(?:also\s+)?'
                rf'(?=(?P<explicit>(?:it|(?:the|my)\s+(?:{subject}))\s+(?:must|should|needs?\s+to)\b|(?:{_TARGET_REQUEST_PATTERN}))'
                rf'|(?:serv(?:e|es|ing)|offer(?:s|ing)?|shar(?:e|es|ing))\b|with\s+{_CUISINE_LIST_PATTERN}\s+on\s+the\s+menu\b)', p,
            ):
                if attached.start() < reference.start():
                    continue
                if not attached.group('explicit') and not owns_continuation(attached.start()):
                    continue
                continuation = p[attached.end():]
                # A following relative price/rating predicate does not negate
                # an affirmative cuisine predicate in the preceding clause.
                continuation = re.split(
                    r'\bbut\s+(?=(?:is\s+)?(?:rated|priced|costs?)\b)', continuation, maxsplit=1,
                )[0].rstrip(' ,')
                merge(_parse_paragraph_constraints(continuation, slot_type))
            # These relative clauses recover a target subject through an
            # intervening target rating, or coordinated serving/rating
            # predicates after a relative price comparison to a meal.
            for relative in re.finditer(r'\bthat\s+(?:also\s+)?(?=(?:serv(?:e|es|ing)|offer(?:s|ing)?)\b)', p):
                if relative.start() <= reference.start():
                    continue
                continuation = p[relative.end():]
                rating_attachment = re.search(
                    r',\s*with\s+(?:a\s+)?rating\s+within\s+\d+(?:\.\d+)?\s*,\s*$', p[:relative.start()],
                )
                paired_rating = re.match(
                    rf'(?:serv(?:e|es|ing)|offer(?:s|ing)?)\s+{_CUISINE_LIST_PATTERN}(?:\s+(?:cuisines?|food))?\s*,?\s+and\s+is\s+rated\s+(?:higher|lower)\b', continuation,
                )
                shared_subject = (
                    paired_rating and re.search(r'\bwithin\s+(?:\$\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?%)\s+of\s*$', prefix)
                    and re.search(r'\b(?:breakfast|lunch|dinner)$', p[reference.start():relative.start()].rstrip(' ,'))
                    and owns_continuation(relative.start())
                )
                if (rating_attachment and owns_continuation(rating_attachment.start())) or shared_subject:
                    attributes = _parse_paragraph_constraints(continuation, slot_type)
                    if attributes.get('cuisines'):
                        merge({'cuisines': attributes['cuisines']})
        return c

    # Explicit absolute intervals need not include "per person/night". A
    # relative difference interval is not an absolute price/rating bound.
    number = r'\d+(?:,\d{3})*(?:\.\d+)?'
    not_offset = r'\b(?![.,]\d)(?!\s*(?:per\s+(?:person|night)\s*)?(?:more|less|higher|lower|above|below|cheaper)\b)'
    price_patterns = (
        rf'\$\s*({number})\s*(?:[–\-—]+|to)\s*\$?\s*({number}){not_offset}',
        rf'\b(?:between|from)\s+\$\s*({number})\s+(?:and|to)\s*\$?\s*({number}){not_offset}',
        rf'\b(?:cost(?:ing|s)?|pric(?:ed|es?))(?:\s+range(?:\s+of)?)?\s+'
        rf'(?:(?:about|around|approximately|between|from)\s+)?\$?\s*({number})\s*(?:[–\-—]+|and|to)\s*\$?\s*({number}){not_offset}',
        rf'\b(?:between|from)\s+({number})\s+(?:and|to)\s+({number})\s+per\s+(?:person|night){not_offset}',
    )
    for pattern in price_patterns:
        price_m = next((match for match in re.finditer(pattern, p)
                        if not re.search(r'\b(?:within|difference(?:\s+of)?)\s*$', p[:match.start()])), None)
        if price_m:
            c['min_price'] = float(price_m.group(1).replace(',', ''))
            c['max_price'] = float(price_m.group(2).replace(',', ''))
            break
    # An approximate single price does not imply an invented numeric range.
    if 'min_price' not in c and re.search(r'cost.*(?:more|less|higher|lower|within).*than', p):
        c['has_relative_constraint'] = True

    # Rating intervals are explicit; relative ratings remain for the LLM.
    rating_m = re.search(
        rf'rat(?:ing|ed)\s+(?:between|of|around|from)\s+({number})\s*(?:and|to|[–\-—])\s*({number}){not_offset}', p,
    ) or re.search(
        rf'\brated\s+({number})\s*(?:to|[–\-—])\s*({number}){not_offset}', p,
    )
    if rating_m:
        c['min_rating'] = float(rating_m.group(1).replace(',', ''))
        c['max_rating'] = float(rating_m.group(2).replace(',', ''))
    # "rated higher/lower/at least" — relative, mark for LLM check
    if 'min_rating' not in c and re.search(r'rat(?:ing|ed).*(?:higher|lower|within|at least)', p):
        c['has_relative_constraint'] = True

    # Known cuisine names can be predicates, menu entries, or modifiers of
    # the requested venue/meal. Cafe and Bakery can also name the venue type.
    cuisine_texts = []
    if re.search(r'\b(?:serv(?:e|ing|es)|offer(?:s|ing)?)\b|\bcuisines?\b', p):
        cuisine_texts.append(p)
    cuisine_phrases = (
        rf'\b({_CUISINE_LIST_PATTERN})\s+(?:food\s+)?(?:place|spot|restaurant|meal|breakfast|lunch|dinner)\b',
        rf'\b({_CUISINE_LIST_PATTERN})\s+on\s+the\s+menu\b',
        r'\b(?:a|an|at)\s+((?:cafe|bakery)(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)(?:cafe|bakery))*)\b',
    )
    for pattern in cuisine_phrases:
        cuisine_texts.extend(match.group(1) for match in re.finditer(pattern, p))
    found_cuisines = [
        cuisine for cuisine in _KNOWN_CUISINES
        if any(re.search(rf'\b{re.escape(cuisine.lower())}\b', phrase) for phrase in cuisine_texts)
    ]
    if found_cuisines:
        c['cuisines'] = found_cuisines
    if c.get('cuisines') and re.search(_NO_OTHER_CUISINES, p):
        exact = re.search(rf'\b(?P<names>{_CUISINE_LIST_PATTERN})\s+cuisines?\s+{_NO_OTHER_CUISINES}', p)
        # An unknown cuisine in the literal list must not silently disappear
        # from an exact-set requirement (e.g. "Japanese and Seafood").
        if exact and not re.search(r'(?:,|&|\b(?:and|or))\s*$', p[:exact.start()]):
            names = {name.lower() for name in re.findall(_CUISINE_NAME_PATTERN, exact.group('names'))}
            c['cuisines'] = [name for name in _KNOWN_CUISINES if name.lower() in names]
            c['cuisines_exact'] = True

    # Room type
    if re.search(r'private\s+room', p):
        c['room_type'] = 'Private room'
    elif re.search(r'entire\s+(?:home|room|apartment|house|home/apt|home\s+or\s+apartment)', p):
        c['room_type'] = 'Entire home/apt'
    elif re.search(r'shared\s+room', p):
        c['room_type'] = 'Shared room'

    # House rules
    for rule, pattern in _HOUSE_RULE_PATTERNS.items():
        if re.search(pattern, p):
            c[rule] = True

    # Required group capacity is a lower bound on the option's max_occupancy.
    # "Up to N people" requires at least that capability; its upper-bound
    # interpretation remains in the original requirement for the LLM.
    if slot_type in (None, 'accommodation'):
        count = rf"(?:\d+|{'|'.join(_CARDINAL_MAP)})"
        count_end = r'(?=\s*(?:[,;!?]|$|\b(?:and|with|that|which)\b))'
        capacity_patterns = (
            rf'\b(?:for|fits?|host|accommodat(?:e|es|ing))\s+(?:up\s+to\s+)?({count})\s+(?:people|persons?|guests?)\b',
            rf'\b(?:fits?|suitable\s+for)\s+({count})\b{count_end}',
            rf'\bprivate\s+room\s+for\s+({count})\b{count_end}',
        )
        for pattern in capacity_patterns:
            for match in re.finditer(pattern, p):
                token = match.group(1)
                size = int(token) if token.isdigit() else _CARDINAL_MAP[token]
                if size > 0:
                    c['min_occupancy'] = max(c.get('min_occupancy', 0), size)
    # Keep an explicitly stated capacity cap distinct from required group size.
    occ_m = re.search(r'\bmaximum\s+occupancy\s+(?:of\s+|is\s+)?(\d+)\b', p)
    if occ_m:
        c['max_occupancy'] = int(occ_m.group(1))

    return c


def _constraint_clause_spans(text: str) -> list[tuple[int, int]]:
    """Keep source offsets at sentence boundaries, not decimals/abbreviations."""
    clauses: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(
        r'\([^()]*\)|\b(?:st|mt|dr|mr|mrs|ms|e\.g|i\.e)\.|\b(?:[a-z]\.){2,}'
        r'|(?P<boundary>[!?;\n]|\.(?=\s|$))', text, re.IGNORECASE,
    ):
        if m.group('boundary'):
            clauses.append((start, m.start()))
            start = m.end()
    clauses.append((start, len(text)))
    return clauses


def _split_constraint_clauses(text: str) -> list[str]:
    """Split sentence/semicolon boundaries, not decimals or abbreviated names."""
    return [text[start:end] for start, end in _constraint_clause_spans(text)]


def _scope_constraint_clause(text: str) -> str:
    """Remove clearly separate background statements from a request clause."""
    if not _extract_day_slot(text):
        return text

    def unrelated(fragment: str) -> bool:
        # Never discard an alternative/negation: it may qualify the request.
        return (_has_independent_subject(fragment) and not _extract_day_slot(fragment)
                and not _has_ambiguous_constraints(fragment))

    def scope_parenthesis(match: re.Match[str]) -> str:
        # A local anaphor can repeat the requested object as its subject. Match
        # it to the attachment, not a global restaurant/hotel subject whitelist.
        prefix = text[:match.start()]
        antecedent = re.search(r'\b(restaurant|hotel|room|accommodation)\s*$', prefix, re.IGNORECASE)
        subject = re.match(r'\s*(?:it\b|the\s+(\w+)\b)', match.group(1), re.IGNORECASE)
        owned = (antecedent and subject
                 and (subject.group(1) is None or subject.group(1).lower() == antecedent.group(1).lower()))
        return ' ' if not owned and unrelated(match.group(1)) else match.group()

    scoped = re.sub(r'\(([^()]*)\)', scope_parenthesis, text)
    # Attribute lists and owned parentheses stay attached; numeric commas are
    # not clause boundaries. A new subject on either side of a comma is not an
    # attribute of the requested restaurant/hotel merely by sharing a sentence.
    fragments: list[str] = []
    start = 0
    for m in re.finditer(r'\([^()]*\)|(?P<boundary>,(?!\d))', scoped):
        if m.group('boundary'):
            fragments.append(scoped[start:m.start()])
            start = m.end()
    fragments.append(scoped[start:])
    return ','.join(fragment for fragment in fragments if not unrelated(fragment))


def _request_sections(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """Split at distinct targets, leaving reference text with its own request."""
    sections: list[tuple[int, list[tuple[int, str]]]] = []
    spans = _constraint_clause_spans(text)
    previous_end = 0
    for start, end, keys in _day_slot_groups(text):
        if sections and sections[-1][1] == keys:
            previous_end = max(previous_end, end)
            continue
        clause_start = next(left for left, right in spans if left <= start <= right)
        if not sections or clause_start >= previous_end:
            start = clause_start
        else:
            # Include an adjoining request verb/preposition instead of leaving
            # "and choose" attached to the preceding target's attributes.
            intro = re.search(
                r'(?:^|,\s*|\band\s+|\bthen\s+)(?P<intro>(?:please\s+)?(?:choose|select|find|book|keep|retain|reuse)\s+(?:(?:a|an|the)\s+)?|(?:on|for)\s+(?:the\s+)?)$',
                text[previous_end:start], re.IGNORECASE,
            )
            if intro:
                start = previous_end + intro.start('intro')
        if sections and start <= sections[-1][0]:
            sections[-1][1].extend(key for key in keys if key not in sections[-1][1])
        else:
            sections.append((start, list(keys)))
        previous_end = max(previous_end, end)
    return [
        (text[start:sections[index + 1][0] if index + 1 < len(sections) else len(text)].strip(), keys)
        for index, (start, keys) in enumerate(sections)
    ]


def _requirement_text(text: str) -> str:
    """Keep the original requirement in one line, without a duplicate label."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+\b(?:and|then)\s*$', '', text, flags=re.IGNORECASE).rstrip(' ,;')
    label, colon, remainder = text.partition(':')
    if colon and remainder.strip() and _extract_day_slot(label):
        plain_label = _DAY_PATTERN.sub('', label.lower())
        plain_label = re.sub(rf'\b(?:{_SLOT_PATTERN})\b', '', plain_label)
        plain_label = re.sub(r'\b(?:the|on|for|and|plus)\b|[\s,&-]', '', plain_label)
        if not plain_label:
            text = remainder.strip()
    return text


def _owned_attribute_clause(text: str, slot: str) -> bool:
    """Recognize an explicit follow-up about the active target's attributes."""
    subject = 'accommodation|hotel|room' if slot == 'accommodation' else 'restaurant'
    return bool(re.match(
        rf'\s*(?:and\s+)?(?:(?:(?:the|its|a)\s+)?(?:nightly\s+(?:rate|cost)|price|rating|cuisines?|room\s+(?:type|category)|maximum\s+occupancy|house\s+rules?|rules)\b'
        rf'|(?:it|(?:the|my)\s+(?:{subject}))\s+(?:must|should|needs?|has|serves|costs|is)\b)',
        text, re.IGNORECASE,
    ))


def _retains_base(text: str, key: tuple[int, str], base_traveler_name: str | None) -> bool:
    """Accept only an explicit, wholly unchanged same-item base arrangement."""
    p = _normalized_query_text(text)
    p = re.sub(r"\b(i|we)'d\b", r'\1 would', p)
    p = re.sub(r"\b(i|we)'ll\b", r'\1 will', p)
    p = re.sub(r"\bi'm\b", 'i am', p)
    p = re.sub(r"\bwe're\b", 'we are', p)
    base_pattern = r"\b(?:the\s+)?base(?:\s+traveler)?(?:'s)?(?:\s+plan)?(?:'s)?\b"
    if base_traveler_name and base_traveler_name.strip():
        name = re.escape(base_traveler_name.strip().lower().replace('’', "'"))
        base_pattern += rf"|(?<!\w){name}(?:'s)?(?!\w)"
    base = re.search(base_pattern, p)
    identity = re.search(
        r'\b(?:keep|retain|reuse|use|leave)\b|\b(?:same|identical)\s+(?:restaurant|hotel|accommodation|arrangement)\b', p,
    )
    item = r'(?:breakfast|lunch|dinner|accommodation|restaurant|hotel|room|meal|place)'
    item_list = rf'{item}(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*){item})*'
    joining_pattern = (
        rf'\b(?:join(?:ing)?\s+|stay(?:ing)?\s+with\s+|shar(?:e|ing)\s+with\s+|eat(?:ing)?\s+with\s+)(?:{base_pattern})'
        rf'|\bstay(?:ing)?\s+(?:at|in)\s+(?:the\s+)?same\s+(?:place|accommodation|hotel)\s+as\s+(?:{base_pattern})'
        rf'|\b(?:shar(?:e|ing)|hav(?:e|ing)|eat(?:ing)?)\s+(?:(?:a|an|the|same|both)\s+)*{item_list}'
        rf'(?:\s+(?:on|for)\s+(?:the\s+)?{_DAY_PATTERN.pattern})?\s+with\s+(?:{base_pattern})'
    )
    # A local "join Rowan for lunch" is an identity request. An opening
    # "joining Rowan for this trip" has no target in its own statement and
    # must not imply that other requested items can all be inherited.
    local_join = False
    for clause in _split_constraint_clauses(p):
        mentions = [
            start + mention.start()
            for start, end, keys in _day_slot_groups(clause) if key in keys
            for mention in re.finditer(rf'\b(?:{_SLOT_PATTERN})\b', clause[start:end])
            if _SLOT_SYNONYMS.get(re.sub(r'\s+', ' ', mention.group()), mention.group()) == key[1]
        ]
        for joining in re.finditer(joining_pattern, clause):
            # A target before/inside the identity phrase belongs to it. A
            # later target needs a local "for/on ..." link, not "and I need".
            local_join |= any(
                position <= joining.end() or re.fullmatch(
                    r'[\s,]*(?:(?:for|on|at|the|both)\s+)+',
                    _DAY_PATTERN.sub('', clause[joining.end():position]),
                ) is not None
                for position in mentions
            )
    if not base or not (identity or local_join):
        return False
    # The identity and base must be in the same statement, not unrelated
    # mentions from different sentences in this target's requirement text.
    if not local_join and not any(
        re.search(base_pattern, clause) and re.search(identity.re.pattern, clause)
        for clause in _split_constraint_clauses(p)
    ):
        return False
    day, slot = key
    groups = _day_slot_groups(p)
    direct_keep = re.match(r'\s*(?:please\s+)?(?:keep|retain|reuse|use|leave)\b', p)
    coordinated_keep = (direct_keep or local_join) and len(groups) == 1 and key in groups[0][2]
    scope = p[:groups[0][0]] + p[groups[0][1]:] if coordinated_keep else p
    days = {day for match in _DAY_PATTERN.finditer(scope) for day in _days_in_mention(match)}
    slots = {_SLOT_SYNONYMS.get(word, word) for word in re.findall(rf'\b(?:{_SLOT_PATTERN})\b', scope)}
    if days - {day} or slots - {slot}:
        return False
    if not local_join and not re.search(r'\b(?:keep|retain|reuse|use|leave)\b', p):
        # "Same restaurant" alone does not identify which of the base's meals
        # it refers to. Require the same explicit reference day and slot.
        reference = p[base.end():]
        ref_days = {value for match in _DAY_PATTERN.finditer(reference) for value in _days_in_mention(match)}
        ref_slots = {_SLOT_SYNONYMS.get(word, word) for word in re.findall(rf'\b(?:{_SLOT_PATTERN})\b', reference)}
        if not ref_days and re.search(r'\bsame\s+day\b', reference):
            ref_days = {day}
        if ref_days != {day} or ref_slots != {slot}:
            return False
    # A narrow identity-only vocabulary intentionally rejects extra conditions
    # (including ones this numeric/categorical parser cannot understand).
    rest = re.sub(base_pattern, '', p)
    rest = _DAY_PATTERN.sub('', rest)
    # Do not erase typed-room synonyms here: "private/shared room" can add a
    # new attribute even when the surrounding request says to join the base.
    rest = re.sub(r'\b(?:breakfast|lunch|dinner|accommodation|stay|lodge|hotel)\b', '', rest)
    if local_join:
        rest = re.sub(r'\b(?:join|joining|staying|share|sharing|with|both|eating|having|room|place|hope|hoping|also|again|well)\b', '', rest)
    rest = re.sub(
        r"\b(?:i|we|would|will|am|are|be|like|want|need|prefer|to|please|a|an|the|at|as|for|on|in|from|of|my|same|identical|exact|exactly|unchanged|original|existing|keep|retain|reuse|use|leave|restaurant|hotel|accommodation|arrangement|plan|traveler|base|chose|chosen|selected|booked|had|previously|eat|have|it|that|and|item|meal|day|night|book|choose|select)\b|(?<=\w)'s\b", '', rest,
    )
    return not re.sub(r"[\s:;,.!?()'–—-]", '', rest)


def _shared_rule_reference(
    text: str,
    day: int,
    prior_queries: Mapping[str, str],
) -> tuple[str, int, str] | None:
    """Resolve the named accommodation immediately before a shared-rule clause."""
    names = '|'.join(re.escape(name).replace(r'\ ', r'\s+')
                     for name in sorted(prior_queries, key=len, reverse=True))
    if not names:
        return None
    owners = list(re.finditer(rf"(?<![\w'-])(?P<name>{names})'s\b", text))
    if not owners:
        return None
    owner = owners[-1]
    prefix = text[:owner.start()]
    comparison = re.search(r'\b(?:of|than|as|to)\s*$', prefix)
    if not comparison or _has_ambiguous_constraints(prefix[:comparison.start()]):
        return None
    suffix = text[owner.end():].strip()
    dates = list(_DAY_PATTERN.finditer(suffix))
    if dates:
        if len(dates) != 1:
            return None
        days = _days_in_mention(dates[0])
        if len(days) != 1:
            return None
        day = days[0]
    # A bare possessive inherits this clause's accommodation/day, never an
    # arbitrary date from the referenced traveler's other requirements.
    room = ' '.join(_DAY_PATTERN.sub('', suffix).split())
    room = re.sub(r'^the\s+', '', room)
    room = re.sub(r'\s+(?:on|for)$', '', room)
    if room and room not in ('accommodation', 'room') and _SLOT_SYNONYMS.get(room) != 'accommodation':
        return None
    return ' '.join(owner.group('name').split()), day, prefix[:comparison.start()]


def _prior_shared_rule_relation(
    prior_queries: Mapping[str, str],
    traveler_name: str,
    day: int,
    pattern: str,
) -> bool | None:
    """Read one public requirement, not option facts or transitive rule flags."""
    # No prior context on this parse: it supplies target/source ownership only
    # and cannot recurse through other travelers' inferred rule attributes.
    targets = _parse_query_slot_constraints(prior_queries[traveler_name])
    requirement = targets.get((day, 'accommodation'))
    if requirement is None:
        return False  # The complete query contains no request for this item.
    p = _normalized_attribute_text(requirement['requirement_text'])
    relations = list(re.finditer(
        rf'\b(?:shar(?:e|es|ing)\s+(?:the\s+)?(?:same\s+)?'
        rf'|(?:has|have|follows?|enforces?)\s+(?:the\s+)?same\s+'
        rf'|same\s+){pattern}\s+rules?\b', p,
    ))
    if not relations:
        return False
    keys = _extract_day_slot(p)
    if ((keys and keys != [(day, 'accommodation')])
            or re.search(r'\b(?:alternatively|instead|either|neither|optional)\b|\bnot\s+(?:required|necessary|mandatory)\b', p)):
        return None
    for shared in relations:
        attachment = re.search(r'(?:,\s*(?:and\s+)?|\band\s+)(?:also\s+)?$', p[:shared.start()])
        if not attachment:
            continue
        reference = _shared_rule_reference(p[:attachment.start()], day, prior_queries)
        if reference is None:
            continue
        other_name, other_day, prefix = reference
        if (other_name, other_day) == (traveler_name, day):
            continue
        if not _has_ambiguous_constraints(prefix + ' ' + p[shared.end():]):
            return True
    # A sharing phrase without a reliably owned, dated third object does not
    # establish either reading. Do not guess from a prior rule's Boolean value.
    return None


def _contextual_shared_house_rules(
    text: str,
    day: int,
    prior_queries: Mapping[str, str],
) -> dict[str, bool]:
    """Resolve only bare named-rule attachment after a named accommodation."""
    p = _normalized_attribute_text(text)
    candidates = [
        (rule, pattern, shared)
        for rule, pattern in _HOUSE_RULE_PATTERNS.items()
        for shared in re.finditer(
            rf'\b(?:that\s+shares\s+(?:the\s+)?(?:same\s+)?|with\s+(?:the\s+)?same\s+)'
            rf'{pattern}\s+rules?\b', p,
        )
    ]
    if not candidates:
        return {}
    queries: dict[str, str] = {}
    for name, query in prior_queries.items():
        name = ' '.join(name.lower().replace('’', "'").split())
        if not name or name == 'traveler' or name in queries or not query.strip():
            return {}
        queries[name] = query
    rules: dict[str, bool] = {}
    for rule, pattern, shared in candidates:
        reference = _shared_rule_reference(p[:shared.start()], day, queries)
        if reference is None:
            continue
        name, reference_day, prefix = reference
        if _has_ambiguous_constraints(prefix + ' ' + p[shared.end():]):
            continue
        if _prior_shared_rule_relation(queries, name, reference_day, pattern) is False:
            rules[rule] = True
    return rules


def _parse_query_slot_constraints(
    query_text: str,
    base_traveler_name: str | None = None,
    *,
    prior_traveler_queries: Mapping[str, str] | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Map target ``(day, slot)`` to attributes and original requirement text.

    ``retain_base`` is true only for clear unchanged same-day/same-item reuse.
    References remain in ``requirement_text`` but never become extra targets.
    ``prior_traveler_queries`` maps unique public names to their full original
    queries, including the base traveler and excluding current/future travelers.
    Missing context leaves bare shared-rule attachment conservatively unparsed.
    """
    result: dict[tuple[int, str], dict[str, Any]] = {}
    source_text: dict[tuple[int, str], list[str]] = {}
    paragraphs = re.split(r'\n\s*\n', query_text.strip())
    for para in paragraphs:
        for section, keys in _request_sections(para):
            constraints: dict[str, Any] = {}
            # A shared list keeps its targets and text, but ambiguous attribute
            # ownership across different targets does not create hard checks.
            if len(keys) == 1:
                owned_context = False
                alternative = False
                for clause in _split_constraint_clauses(section):
                    if not clause.strip():
                        continue
                    clause_keys = _extract_day_slot(clause)
                    if clause_keys == keys:
                        owned_context = True
                    elif clause_keys or not owned_context or not _owned_attribute_clause(clause, keys[0][1]):
                        # A later alternative can qualify an earlier sentence;
                        # it is not safe to keep that sentence's hard checks.
                        if re.search(r'\b(?:alternatively|instead|either|neither|optional)\b|^\s*(?:or|not|but|unless)\b|\bnot\s+(?:required|necessary|mandatory)\b', clause, re.IGNORECASE):
                            alternative = True
                        owned_context = False
                        continue
                    scoped = _scope_constraint_clause(clause)
                    constraints.update(_parse_paragraph_constraints(scoped, keys[0][1]))
                    if keys[0][1] == 'accommodation' and prior_traveler_queries is not None:
                        constraints.update(_contextual_shared_house_rules(scoped, keys[0][0], prior_traveler_queries))
                if alternative:
                    constraints = {name: value for name, value in constraints.items()
                                   if name in ('is_join', 'has_relative_constraint')}
            for key in keys:
                result.setdefault(key, {}).update(constraints)
                source_text.setdefault(key, []).append(section)
    for key, constraints in result.items():
        sections = source_text[key]
        constraints['requirement_text'] = ' '.join(dict.fromkeys(_requirement_text(section) for section in sections))
        # All requirements for a repeated target must permit unchanged reuse.
        constraints['retain_base'] = all(_retains_base(section, key, base_traveler_name) for section in sections)
    return result


def _check_option_against_constraints(
    option: dict[str, Any],
    constraints: dict[str, Any],
    slot_type: str,
) -> list[str]:
    """Check a selected option against parsed constraints.

    Returns a list of human-readable violation strings (empty = all OK).
    """
    violations: list[str] = []

    # --- Price ---
    min_p = constraints.get('min_price')
    max_p = constraints.get('max_price')
    if min_p is not None or max_p is not None:
        actual_price = to_float(option.get('price'))
        if actual_price is not None:
            if min_p is not None and actual_price < min_p:
                violations.append(f"price ${actual_price} < min ${min_p}")
            if max_p is not None and actual_price > max_p:
                violations.append(f"price ${actual_price} > max ${max_p}")

    # --- Rating ---
    min_r = constraints.get('min_rating')
    max_r = constraints.get('max_rating')
    if min_r is not None or max_r is not None:
        actual_rating = to_float(option.get('rating'))
        if actual_rating is not None:
            if min_r is not None and actual_rating < min_r:
                violations.append(f"rating {actual_rating} < min {min_r}")
            if max_r is not None and actual_rating > max_r:
                violations.append(f"rating {actual_rating} > max {max_r}")

    # --- Cuisines ---
    required_cuisines = constraints.get('cuisines')
    actual_cuisines = option.get('cuisines')
    if (required_cuisines and slot_type in ('breakfast', 'lunch', 'dinner')
            and isinstance(actual_cuisines, list)):
        actual_set = {c.strip().lower() for c in actual_cuisines}
        for req in required_cuisines:
            if req.strip().lower() not in actual_set:
                violations.append(f"missing cuisine '{req}'")
        if constraints.get('cuisines_exact'):
            required_set = {c.strip().lower() for c in required_cuisines}
            extras = sorted(actual_set - required_set)
            if extras:
                violations.append(f"unexpected cuisine types: {', '.join(extras)} (no other cuisine types permitted)")

    # --- Room type ---
    req_room = constraints.get('room_type')
    if req_room and slot_type == 'accommodation':
        actual_room = option.get('room_type')
        if (isinstance(actual_room, str) and actual_room.strip() not in ('', '-')
                and actual_room.strip().lower() != req_room.lower()):
            violations.append(f"room_type='{actual_room}' but requires '{req_room}'")

    # --- House rules ---
    actual_rules = option.get('rules')
    if slot_type == 'accommodation' and isinstance(actual_rules, list):
        for rule, pattern in _HOUSE_RULE_PATTERNS.items():
            if constraints.get(rule) and not any(
                re.search(pattern, value.lower().replace('’', "'")) for value in actual_rules
            ):
                label = rule.replace('_', ' ').capitalize()
                violations.append(f"requires '{label}' rule but not present")

    # --- Accommodation capacity ---
    req_min_occ = constraints.get('min_occupancy')
    req_max_occ = constraints.get('max_occupancy')
    if slot_type == 'accommodation' and (req_min_occ is not None or req_max_occ is not None):
        actual_occ = to_float(option.get('max_occupancy'))
        if actual_occ is not None:
            if req_min_occ is not None and actual_occ < req_min_occ:
                violations.append(f"max_occupancy={actual_occ} < required capacity {req_min_occ}")
            if req_max_occ is not None and actual_occ > req_max_occ:
                violations.append(f"max_occupancy={actual_occ} > required {req_max_occ}")

    return violations


def _build_selection_map(
    plan: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Map ``(day, slot)`` directly to the plan's selected option dictionaries."""
    return {
        (entry['days'], slot): option
        for entry in plan
        for slot, option in entry.get('selected_options', {}).items()
    }


def _expected_city_for_slot(
    base_plan: list[dict[str, Any]],
    day: int,
    slot: str,
) -> str | None:
    """Read an explicit city from the base's same day and slot, if available."""
    for entry in base_plan:
        if entry['days'] != day:
            continue
        option = entry.get('selected_options', {}).get(slot, {})
        city = option.get('city')
        if isinstance(city, str) and city.strip() not in ('', '-'):
            return city.strip()
        # Supplied base slots use "<name>, <City>" even without selected_options.
        value = entry.get(slot)
        if isinstance(value, str) and ',' in value:
            name, city = value.rsplit(',', 1)
            if name.strip() and city.strip() not in ('', '-'):
                return city.strip()
        return None
    return None


def verify_slot_constraints(
    query_text: str,
    plan: list[dict[str, Any]],
    base_plan: list[dict[str, Any]] | None = None,
    base_traveler_name: str | None = None,
    *,
    prior_traveler_queries: Mapping[str, str] | None = None,
) -> list[str]:
    """Return definite violations in the plan's current selected options.

    An empty list means no deterministic violation was found. Missing required
    slots and complex conditions are checked by the caller.
    """
    slot_constraints = _parse_query_slot_constraints(
        query_text, base_traveler_name, prior_traveler_queries=prior_traveler_queries,
    )
    selection_map = _build_selection_map(plan)
    all_violations: list[str] = []

    if base_plan:
        for day, slot in slot_constraints:
            option = selection_map.get((day, slot))
            if option is None:
                continue
            expected_city = _expected_city_for_slot(base_plan, day, slot)
            actual_city = option.get('city')
            if (not expected_city or not isinstance(actual_city, str)
                    or actual_city.strip() in ('', '-')):
                continue
            # City-state suffixes such as "Toledo(Ohio)" are formatting only.
            if normalize_city(actual_city) != normalize_city(expected_city):
                selected_name = option.get('name', '?')
                all_violations.append(
                    f"Day {day} {slot} '{selected_name}' is in '{actual_city}' "
                    f"but the base plan's {slot} is in '{expected_city}' on day {day}"
                )

    for (day, slot), constraints in slot_constraints.items():
        option = selection_map.get((day, slot))
        if option is None:
            continue  # Unfilled slots are handled by the caller.
        violations = _check_option_against_constraints(option, constraints, slot)
        if violations:
            selected_name = option.get('name', '?')
            all_violations.append(
                f"Day {day} {slot} '{selected_name}': {'; '.join(violations)}"
            )

    return all_violations
