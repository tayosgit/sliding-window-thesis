class CommunicationRequest:  # TODO Überprüfen, ob es vielleicht Sinn ergibt statt ints node Instanzen zu speichern
    def __init__(self, id, src, dst):
        self.id = id
        self.src = src
        self.dst = dst
        # self.cost = 0

    # def __str__(self):
    #     return f"CommunicationRequest(src={self.src}, dst={self.dst})"

    def __repr__(self):
        return f"CR(src={self.src}, dst={self.dst})"

    def get_source(self):
        return self.src

    def get_destination(self):
        return self.dst

    # def get_size(self):
    #     return self.size

    # Fulfilment Cost muss noch berechnet werden
    # def set_cost(self, c):
    #     self.cost = c
