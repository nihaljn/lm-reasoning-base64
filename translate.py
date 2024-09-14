import base64


def english_to_base64(text: str) -> str:
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def base64_to_english(base64_text: str) -> str:
    return base64.b64decode(base64_text.encode('utf-8')).decode('utf-8')


if __name__ == "__main__":
    english_str = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Bob writes down a number between 1 and 1,000. Mary must identify that number by asking "yes/no" questions of Bob. Mary knows that Bob always tells the truth. If Mary uses an optimal strategy, then she will determine the answer at the end of exactly how many questions in the worst case?

A) 1,000
B) 999
C) 500
D) 10"""
    base64_str = """QW5zd2VyIHRoZSBmb2xsb3dpbmcgbXVsdGlwbGUgY2hvaWNlIHF1ZXN0aW9uLiBUaGUgbGFzdCBsaW5lIG9mIHlvdXIgcmVzcG9uc2Ugc2hvdWxkIGJlIG9mIHRoZSBmb2xsb3dpbmcgZm9ybWF0OiAnQW5zd2VyOiAkTEVUVEVSJyAod2l0aG91dCBxdW90ZXMpIHdoZXJlIExFVFRFUiBpcyBvbmUgb2YgQUJDRC4gVGhpbmsgc3RlcCBieSBzdGVwIGJlZm9yZSBhbnN3ZXJpbmcuCgpCb2Igd3JpdGVzIGRvd24gYSBudW1iZXIgYmV0d2VlbiAxIGFuZCAxLDAwMC4gTWFyeSBtdXN0IGlkZW50aWZ5IHRoYXQgbnVtYmVyIGJ5IGFza2luZyAieWVzL25vIiBxdWVzdGlvbnMgb2YgQm9iLiBNYXJ5IGtub3dzIHRoYXQgQm9iIGFsd2F5cyB0ZWxscyB0aGUgdHJ1dGguIElmIE1hcnkgdXNlcyBhbiBvcHRpbWFsIHN0cmF0ZWd5LCB0aGVuIHNoZSB3aWxsIGRldGVybWluZSB0aGUgYW5zd2VyIGF0IHRoZSBlbmQgb2YgZXhhY3RseSBob3cgbWFueSBxdWVzdGlvbnMgaW4gdGhlIHdvcnN0IGNhc2U/CgpBKSAxLDAwMApCKSA5OTkKQykgNTAwCkQpIDEw"""
    translated_base64 = base64_to_english(base64_str)
    translated_english = english_to_base64(english_str)

    assert base64_to_english(english_to_base64(english_str)) == english_str
    assert english_to_base64(base64_to_english(base64_str)) == base64_str
    assert translated_base64 == english_str
    assert translated_english == base64_str

    print("Passed all tests!")