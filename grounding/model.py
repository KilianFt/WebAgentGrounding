import abc

class GroundingModel(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, demos):
        # return list of samples that contains uid and if it is the label
        pass

    @abc.abstractmethod
    def predict_score(self, samples):
        # return list of scores
        pass

    def predict(self, demos):
        samples = self.preprocess(demos)
        preds = self.predict_score(samples)

        out = []
        for r in preds:
            out.append({
                'uid': r['uid'],
                'score': r['score'],
                'rank': r['rank'],
                'label': r['label']
            })
        return out