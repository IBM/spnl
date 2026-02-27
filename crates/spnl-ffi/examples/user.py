import spnl

x = spnl.from_str('(user "hello")')

match x:
    case spnl.Unit.User(s):
        print(f"User message {s[0]}")
