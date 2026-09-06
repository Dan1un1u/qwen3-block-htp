#!/usr/bin/env python3
"""Independent positive/negative behavioral cases for content grading."""
import unittest
from score_short_tasks_v2 import grade_content


def sample(scorer, answer, prompt="", category="reading"):
    return dict(scorer=scorer, answer=answer, prompt=prompt, category=category)


class ContentScoringTests(unittest.TestCase):
    def check_cases(self, task, cases):
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertIs(grade_content(task, text)["content_correct"], expected)

    def test_numeric_wrappers_and_collisions(self):
        self.check_cases(sample("numeric", 5), [
            ("5", True), ("  +5.000  ", True), ("５。", True),
            ("The answer is 5.", True), ("答案：5", True), ("**5**", True),
            ("15", False), ("-5", False), ("5.1", False),
            ("5 or 15", None), ("Not 5", None), ("5, but actually 15", None),
            ("The question contains 5 pens", None), ("5%", None), ("", None),
            ("5\n6", None), ("5\nThe answer is 15", None),
        ])

    def test_equations_are_verified(self):
        self.check_cases(sample("numeric", 18), [
            ("6×3=18", True), ("6 times 3 is 18.", True),
            ("(12+6)/1=18", True), ("6+3=18", False), ("6*3=6", False),
            ("18=6+12", None), ("6*3=18=19", None),
            ("__import__('os')=18", None), ("6**3=18", None),
            ("1/0=18", None), ("6*3=", None),
        ])
        self.check_cases(sample("numeric", 15), [("7 plus 8 is 15.", True)])
        self.check_cases(sample("numeric", 7), [("12 - 5 = 7", True), ("12-5=5", False)])

    def test_json_structure_and_types(self):
        self.check_cases(sample("json", {"ok": True}), [
            ('{"ok":true}', True), ('```json\n{"ok": true}\n```', True),
            ('{"ok":1}', False), ('{"ok":"true"}', False),
            ('{"ok":false}', False), ('{"ok":true,"extra":0}', False),
            ('[{"ok":true}]', False),
            ('{"ok":false,"ok":true}', None), ('{"ok": NaN}', None),
            ('```json\n{"ok":true}', None),
            ('{"ok":true}\nBut it is false.', None),
        ])
        self.check_cases(sample("json", {"n": 3}), [('{"n":3.0}', False), ('{"n":3}', True)])
        self.check_cases(sample("json", {"word":"CAT"}), [('{"word":"ＣＡＴ"}', False)])

    def test_text_content_without_answer_substrings(self):
        self.check_cases(sample("text", ["Ben"], "Who takes notes?"), [
            ("Ben", True), ("Ben.", True), ("Ben takes notes.", True),
            ("**Ben**", True), ("Answer: Ben", True),
            ("Benny", False), ("Carol takes notes.", False),
            ("Ben or Carol", None), ("Ben does not take notes", None),
            ("Not Ben", None), ("Ben takes notes. No, Carol does.", None),
        ])
        self.check_cases(sample("text", ["小王"], "谁负责记录？"), [
            ("小王负责记录。", True), ("负责记录的是小王。", True),
            ("小林负责记录。", False), ("小王不负责记录", False),
        ])

    def test_transformation_case_is_semantic(self):
        self.check_cases(sample("text", ["CAT"], "Convert cat to uppercase.", "format"), [
            ("CAT.", True), ("cat", False), ("CATE", False),
            ('the word cat converted to uppercase is "CAT"', True),
            ('**cat**  \n→ **CAT**', True),
            ('the word cat converted to uppercase is "cat"', False),
        ])

    def test_unsupported_not_mislabeled_logic(self):
        self.check_cases(sample("text", ["苹果"]), [("苹果\n苹果", True), ("也许是苹果吧", None)])
        self.check_cases(sample("text", ["周四"], "会议最终在哪一天？"), [
            ("根据题目信息，会议最终在周四举行。", True),
            ("会议最终在周二举行。", False),
        ])


if __name__ == "__main__":
    unittest.main()
