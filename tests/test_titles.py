import unittest

from shorts import reaction_title
from titles import TITLE_LIMIT, short_title


def transcript(*lines):
    return [dict(start=index * 5, end=index * 5 + 4, text=text) for index, text in enumerate(lines)]


class TitleTests(unittest.TestCase):
    def test_uses_setup_and_late_reveal_instead_of_greeting(self):
        segments = transcript('Всем привет, ребята!', 'Сейчас попробуем угадать голос.',
                              'Я был уверен, что это голос Папича.',
                              'Оказалось, что это вообще другой человек!',
                              'Ничего себе, как он меня обманул!')
        title = short_title(segments, 0, 25)
        self.assertIn('Папича', title)
        self.assertIn('другой человек', title)
        self.assertNotIn('привет', title)
        self.assertLessEqual(len(title), TITLE_LIMIT)

    def test_keeps_complete_payoff_and_punctuation(self):
        title = short_title(transcript('Давайте посмотрим.', 'Поставил сложность на максимум.',
                                        'С третьей попытки наконец победил босса!',
                                        'Всё, на сегодня хватит.'), 0, 20)
        self.assertIn('победил босса!', title)
        self.assertNotIn('посмотрим', title)
        self.assertNotIn('хватит', title)

    def test_combines_context_and_reaction(self):
        title = short_title(transcript('Ну вот мы снова здесь.', 'Я открыл последний сундук.',
                                        'Там лежало сразу три легендарных меча!',
                                        'Спасибо за просмотр.'), 0, 20)
        self.assertIn('сундук', title)
        self.assertIn('три легендарных меча!', title)
        self.assertNotIn('Спасибо', title)

    def test_long_sentence_uses_complete_clause_not_first_words(self):
        title = short_title(transcript('Он долго пытался разобраться, почему компьютер отказывается '
                                        'включаться, но оказалось, что кот выдернул кабель!'), 0, 5)
        self.assertIn('кот выдернул кабель!', title)
        self.assertNotEqual(title, 'оказалось')
        self.assertLessEqual(len(title), TITLE_LIMIT)

    def test_excludes_words_outside_the_selected_clip(self):
        segments = [dict(start=0, end=20, text='whole segment', words=[
            dict(start=0, end=5, word='Снаружи выиграл миллион!'),
            dict(start=10, end=15, word='С третьей попытки победил босса!'),
            dict(start=16, end=20, word='Снаружи сломал компьютер!')])]
        title = short_title(segments, 10, 15)
        self.assertIn('победил босса', title)
        self.assertNotIn('миллион', title)
        self.assertNotIn('компьютер', title)

    def test_title_survives_word_timing_and_segment_boundaries(self):
        segments = [dict(start=0, end=5, text='', words=[dict(start=0, end=4, word='Я открыл последний')]),
                    dict(start=5, end=10, text='', words=[dict(start=5, end=9, word='сундук.')]),
                    dict(start=10, end=15, text='', words=[dict(start=10, end=14, word='Там оказался легендарный меч!')])]
        title = short_title(segments, 0, 15)
        self.assertIn('Я открыл последний сундук', title)
        self.assertIn('легендарный меч', title)

    def test_empty_and_unpunctuated_transcript(self):
        self.assertEqual(short_title([], 0, 10), 'Момент из стрима')
        title = short_title(transcript('Всем привет ребята', 'Наконец нашёл потерянный ключ'), 0, 10)
        self.assertIn('потерянный ключ', title)
        self.assertNotIn('привет', title)

    def test_longer_title_is_rendered_in_full(self):
        title = 'Я был уверен, что это голос Папича — Оказалось, что это вообще другой человек!'
        overlay = reaction_title(title, 35)
        self.assertIn('ДРУГОЙ ЧЕЛОВЕК!', overlay.replace(r'\N', ' '))
        self.assertIn('DejaVu Sans,52,', overlay)
        self.assertEqual(overlay.count('Dialogue:'), 1)
        self.assertIn('0:00:35.00', overlay)


if __name__ == '__main__':
    unittest.main()
