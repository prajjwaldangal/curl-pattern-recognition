# eager execution vs computational graph
# Inception V3
# abc = "banana"
class foo:
    # abc = ""
    # self.abc = "cake"
    def __init__(self):
        self.abc = "cake"

    def bar(self):

        print(self.abc)
        abc = "apple"
        print(abc)

k = foo().bar()
