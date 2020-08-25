import os


class Individual():
    def __init__(
        self,
        indv_id:int,
        path_to_model:str,
        summary:str,
        parent_id:int,
        bulk_counter:int,
        cut_counter:int,
        bulk_offsprings:int,
        cut_offsprings:int,
        optimizer_config:dict,
        learning_curves:list,
        n_parameters: int,
        ):
        self.indv_id = indv_id
        self.path_to_model = path_to_model
        self.summary = summary
        self.parent_id = parent_id
        self.bulk_counter = bulk_counter
        self.cut_counter = cut_counter
        self.bulk_offsprings = bulk_offsprings
        self.cut_offsprings = cut_offsprings
        self.optimizer_config = optimizer_config
        self.pre_training_loss = learning_curves["validation_loss"][0]
        self.post_training_loss = learning_curves["validation_loss"][-1]
        self.post_training_accuracy = learning_curves["validation_accuracy"][-1]  # We want to optimize this ...
        self.n_parameters = n_parameters  # ... and this.

    def to_dict(self):
        return {
            "id" : self.indv_id,
            "accuracy" : self.post_training_accuracy,
            "n_parameters" : self.n_parameters,
            "parent_id" : self.parent_id,
            "bulk_counter" : self.bulk_counter,
            "cut_counter" : self.cut_counter,
            "bulk_offsprings" : self.bulk_offsprings,
            "cut_offsprings" : self.cut_offsprings,
            "loss_before_training" : self.pre_training_loss,
            "loss_after_training" : self.post_training_loss,
        }

    def __str__(self):
        n_ljust = 25
        thestring = f"Model {self.indv_id}\n\n"
        thestring += self.summary + "\n\n"
        for k, v in self.optimizer_config.items():
            thestring += str(k).ljust(n_ljust) + str(v) + "\n"
        thestring += "\n"
        for k, v in self.to_dict().items():
            thestring += str(k).ljust(n_ljust) + str(v) + "\n"
        return thestring

    def save_info(self):
        if not os.path.exists(self.path_to_model):
            raise Exception("Save model first, then its info")
        with open(self.path_to_model + ".info", "x") as info_file:
            info_file.write(str(self))
