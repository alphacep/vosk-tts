# -- coding: utf-8 --

# Converts an accented vocabulary to dictionary, for example
#
# абстракцион+истов
# абстр+акцию
# абстр+акция
# 
# абстракционистов a0 b s t r a0 k c i0 o0 nj i1 s t o0 v
# абстракцию a0 b s t r a1 k c i0 j u0
# абстракция a0 b s t r a1 k c i0 j a0
#

softletters=set(u"яёюиье")
startsyl=set(u"#ъьаяоёуюэеиы-")
others = set(["#", "+", "-", u"ь", u"ъ"])

softhard_cons = {
    u"б" : u"b",
    u"в" : u"v",
    u"г" : u"g",
    u"Г" : u"g",
    u"д" : u"d",
    u"з" : u"z",
    u"к" : u"k",
    u"л" : u"l",
    u"м" : u"m",
    u"н" : u"n",
    u"п" : u"p",
    u"р" : u"r",
    u"с" : u"s",
    u"т" : u"t",
    u"ф" : u"f",
    u"х" : u"h"
}

other_cons = {
    u"ж" : u"zh",
    u"ц" : u"c",
    u"ч" : u"ch",
    u"ш" : u"sh",
    u"щ" : u"sch",
    u"й" : u"j"
}

vowels = {
    u"а" : u"a",
    u"я" : u"a",
    u"у" : u"u",
    u"ю" : u"u",
    u"о" : u"o",
    u"ё" : u"o",
    u"э" : u"e",
    u"е" : u"e",
    u"и" : u"i",
    u"ы" : u"y",
}                                

def pallatize(phones):
    for i, phone in enumerate(phones[:-1]):
        if phone[0] in softhard_cons:
            if phones[i+1][0] in softletters:
                phones[i] = (softhard_cons[phone[0]] + "j", 0)
            else:
                phones[i] = (softhard_cons[phone[0]], 0)
        if phone[0] in other_cons:
            phones[i] = (other_cons[phone[0]], 0)

def convert_vowels(phones):
    new_phones = []
    prev = ""
    for phone in phones:
        if prev in startsyl:
            if phone[0] in set(u"яюеё"):
                new_phones.append("j")
        if phone[0] in vowels:
            new_phones.append(vowels[phone[0]] + str(phone[1]))
        else:
            new_phones.append(phone[0])
        prev = phone[0]

    return new_phones

def convert(stressword):
    phones = ("#" + stressword + "#")


    # Assign stress marks
    stress_phones = []
    stress = 0
    for phone in phones:
        if phone == "+":
            stress = 1
        else:
            stress_phones.append((phone, stress))
            stress = 0

    # Pallatize
    pallatize(stress_phones)

    # Assign stress
    phones = convert_vowels(stress_phones)

    # Filter
    phones = [x for x in phones if x not in others]

    return " ".join(phones)
