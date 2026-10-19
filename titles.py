"""Extract complete, lively titles from the whole clip without inventing events."""
from collections import Counter
import math
import re

TITLE_LIMIT = 96
_STOP = set('''а и но да или что это как так вот ну же бы ли не на в во по из за для у
с со к ко до от о об то он она они мы вы я ты его ее их мне меня нам вам
тут там уже еще просто очень вообще сегодня сейчас потом все был была было
были будет этот эта эти такой такая такое который когда где если чтобы
the a an and or but is are was were to of in on at for it this that i you we
he she they my your just really now then'''.split())
_FILLER = re.compile(r'^(?:(?:ну|вот|короче|ладно|ребят(?:а)?|в общем|то есть|как бы|'
                     r'на самом деле|так сказать|собственно|в принципе)[,\s]+)+', re.I)
_GENERIC = re.compile(r'^(?:всем привет|привет(?: всем| чат| ребят)?|добро пожаловать|'
                      r'начинаем стрим|сейчас посмотрим|давайте посмотрим|сейчас попробуем|давайте попробуем|ну что|'
                      r'hello everyone|hi everyone|welcome back)\b', re.I)
_ACTION = re.compile(r'\b(?:выигр\w*|проигр\w*|побед\w*|слом\w*|перепут\w*|'
                     r'угад\w*|обман\w*|спас\w*|потер\w*|наш[её]л\w*|нашла|'
                     r'оказал\w*|получил\w*|получилось|сработал\w*|удалил\w*|'
                     r'забыл\w*|застрял\w*|открыл\w*|неожидан\w*|наконец|'
                     r'won|lost|broke|found|saved|failed|surpris\w*)\b', re.I)
_REACTION = re.compile(r'(?:ничего себе|не может быть|вот это|как же|почему|зачем|'
                       r'серь[её]зно|невероятно|офигеть|не ожидал|wow|no way|what a)', re.I)
_LINK = re.compile(r'^(?:а\b|но\b|зато\b|оказал\w*|в итоге\b|в результате\b|'
                   r'теперь\b|тогда\b|там\b|это\b|он\b|она\b|они\b|'
                   r'ничего себе\b|не может быть\b|but\b|then\b|it\b)', re.I)

_REVEAL = re.compile(r'^(?:а\s+)?(?:но\s+)?(?:оказал\w*|в итоге\b|в результате\b|зато\b|но\b)', re.I)


def _content(text):
    return {word for word in re.findall(r'[^\W\d_]+', text.casefold().replace('ё', 'е'))
            if len(word) > 2 and word not in _STOP}


def _context_title(transcript, start, end, limit=TITLE_LIMIT):
    """Rank complete sentences/clauses and connected setup/payoff pairs, not prefixes."""
    parts = []
    for segment in sorted(transcript, key=lambda s: float(s['start'])):
        if float(segment['end']) <= start or float(segment['start']) >= end:
            continue
        words = segment.get('words')
        text = (' '.join(str(word.get('word', '')) for word in words
                         if float(word['end']) > start and float(word['start']) < end)
                if words else str(segment.get('text', '')))
        if text.strip():
            parts.append(text.strip())
    if not parts:
        return 'Момент из стрима'

    # Keep punctuation and full sentences, even when Whisper split one sentence
    # across several segments. Unpunctuated segments remain useful candidates.
    text = ' '.join(parts)
    sentences = [(match.group(), match.start()) for match in re.finditer(r'[^.!?…]+(?:[.!?…]+|$)', text)]
    raw = list(sentences)
    if len(sentences) < 2:
        cursor = 0
        for part in parts:
            raw.append((part, cursor))
            cursor += len(part) + 1
    for sentence, offset in sentences:
        if len(sentence.strip().upper()) > limit:
            clauses = re.split(r'[,;:]\s+(?!(?:что|как|почему|когда|чтобы|котор\w*|потому)\b)|\s+[—–]\s+|\s+(?=но\b|зато\b|в итоге\b)', sentence, flags=re.I)
            cursor = 0
            for clause in clauses:
                position = sentence.find(clause, cursor)
                raw.append((clause, offset + position))
                cursor = position + len(clause)

    phrases, seen = [], set()
    for phrase, position in raw:
        phrase = _FILLER.sub('', ' '.join(phrase.split())).strip(' ,;:—–-.…')
        phrase = re.sub(r'([!?])[!?]+', r'\1', phrase)
        key = phrase.casefold().rstrip('!?')
        if not phrase or key in seen or len(phrase.upper()) > limit:
            continue
        seen.add(key)
        tokens = re.findall(r'[^\W\d_]+', phrase)
        content = _content(phrase)
        if not content or _GENERIC.search(phrase) or (len(tokens) < 2 and not _REACTION.search(phrase)):
            continue
        phrases.append(dict(text=phrase, words=len(tokens), content=content, position=position))
    if not phrases:
        # Prefer a complete short segment to an arbitrary first-N-words cut.
        short = next((part for part in reversed(parts) if len(part.upper()) <= limit and not _GENERIC.search(part)), None)
        return short or 'Момент из стрима'

    phrases.sort(key=lambda phrase: phrase['position'])
    frequencies = Counter(word for part in parts for word in _content(part))
    max_position = max(1, len(text))
    for phrase in phrases:
        content = phrase['content']
        topicality = sum(math.log1p(frequencies[word]) for word in content) / math.sqrt(len(content))
        phrase['energy'] = min(2., len(_ACTION.findall(phrase['text'])) * .8
                               + bool(_REACTION.search(phrase['text'])) * .8
                               + ('!' in phrase['text']) * .3 + ('?' in phrase['text']) * .2)
        phrase['score'] = (topicality + phrase['energy'] + min(phrase['words'], 12) * .07
                           + .2 * phrase['position'] / max_position + 1.1 * bool(_REVEAL.search(phrase['text']))
                           - max(0, phrase['words'] - 16) * .15)
        if phrase['words'] < 3:
            phrase['score'] -= 1.2
        if _LINK.search(phrase['text']):
            phrase['score'] -= .35  # Such a phrase often needs its setup.

    candidates = [(phrase['score'], phrase['position'], phrase['text']) for phrase in phrases]
    # A connected setup + payoff can convey more of the clip than either quote.
    for index, setup in enumerate(phrases):
        if setup['words'] < 3 or _LINK.search(setup['text']):
            continue
        for payoff in phrases[index + 1:]:
            if payoff['position'] <= setup['position'] or payoff['energy'] < .3:
                continue
            shared = setup['content'] & payoff['content']
            if not shared and not _LINK.search(payoff['text']):
                continue
            if setup['text'] in payoff['text'] or payoff['text'] in setup['text']:
                continue
            combined = setup['text'].rstrip('!?') + ' — ' + payoff['text']
            if len(combined.upper()) > limit or setup['words'] + payoff['words'] > 18:
                continue
            reveal = bool(_REVEAL.search(payoff['text']))
            score = (.55 * setup['score'] + .65 * payoff['score'] + .8
                     + min(.45, len(shared) * .15) + reveal)
            candidates.append((score, payoff['position'], combined))
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


_PRESENT = {
    'думаю': 'думает', 'считаю': 'считает', 'помню': 'помнит', 'понимаю': 'понимает',
    'знаю': 'знает', 'смотрю': 'смотрит', 'играю': 'играет', 'пытаюсь': 'пытается',
    'хочу': 'хочет', 'рассказываю': 'рассказывает', 'вспоминаю': 'вспоминает',
    'могу': 'может', 'буду': 'будет', 'угадываю': 'угадывает', 'выбираю': 'выбирает',
}
_TOPIC = re.compile(r'^(?:я\s+)?(?:расскажу|рассказывал|рассказываю|вспоминаю|'
                    r'вспомнил|говорю|говорил|поговорим)\s+(про|о|об)\s+(.+)$', re.I)
_PAST = re.compile(r'^(?:выиграл|проиграл|победил|сломал|перепутал|угадал|обманул|'
                   r'спас|потерял|наш[её]л|получил|удалил|забыл|застрял|открыл|'
                   r'попробовал|увидел|встретил|решил|рассказал|показал)\b', re.I)


def short_title(transcript, start, end, style='По фрагменту'):
    """P52 profile: actor + evidenced event, a topic, or an intact short question."""
    if style not in ('P52', 'По фрагменту'):
        raise ValueError('Неизвестный стиль названия: ' + str(style))
    if style != 'P52':
        return _context_title(transcript, start, end)
    # Reserve room before ranking, so adding the name never cuts off the result.
    base = _context_title(transcript, start, end, limit=TITLE_LIMIT - len('P52: '))
    topic = _TOPIC.match(base)
    person = re.match(r'^я\s+(не\s+)?(\w+)\b(.*)$', base, re.I)
    if topic:
        title = 'P52 ' + topic.group(1) + ' ' + topic.group(2)
    elif person:
        negative, verb, rest = person.groups()
        verb_lower = verb.casefold()
        if re.fullmatch(r'\w+л(?:ся)?', verb_lower) or verb_lower in ('мог', 'смог', 'должен'):
            title = 'P52 ' + (negative or '') + verb + rest
        elif verb_lower in _PRESENT:
            title = 'P52 ' + (negative or '') + _PRESENT[verb_lower] + rest
        else:
            title = 'P52: ' + base
    else:
        # Only use an implicit subject for an actual leading action, not when
        # another person is mentioned. Other people's names are preserved.
        action = re.sub(r'^(?:с\s+\w+\s+(?:попытки|раза)\s+)?(?:наконец\s+)?', '', base, flags=re.I)
        title = ('P52 ' if _PAST.match(action) else 'P52: ') + base
    # The video overlay uses uppercase too; match the reference's title casing.
    upper = title.upper()
    return upper if len(upper) <= TITLE_LIMIT else title
