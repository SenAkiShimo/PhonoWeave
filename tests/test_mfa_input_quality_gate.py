from phonoweave import mfa_input_quality_gate as q

def test_phone_inventory_contracts():
    assert q.phone_contract(['a'],['a˥'])
    assert q.phone_contract(['m','a'],['m','a˧˥'])
    assert q.phone_contract(['{ɻ,ʐ}','u','ŋ'],['ɻ','u˧˥','ŋ'])
    assert not q.phone_contract(['a'],['tɕ','e˥','n'])
    assert not q.phone_contract(['{ɻ,ʐ}','u','ŋ'],['n','u˩','ŋ'])

def test_tone_normalization_does_not_erase_phone_identity():
    assert q.normalize_phone('ʐ̩˥')=='ʐ̩'
    assert q.normalize_phone('a˧˥')=='a'
    assert q.normalize_phone('tɕ')=='tɕ'
