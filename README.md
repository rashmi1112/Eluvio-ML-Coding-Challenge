# Eluvio-ML-Coding-Challenge

Scene‌ ‌Segmentation‌ ‌Using‌ ‌the‌ ‌MovieScenes‌ ‌Dataset‌ ‌

One‌ ‌scene‌ ‌contains‌ ‌a‌ ‌series‌ ‌of‌ ‌consecutive‌ ‌shots.‌ ‌To‌ ‌define‌ ‌the‌ ‌problem,‌ ‌we‌ ‌first‌ ‌carry‌
‌out‌ ‌shot‌ ‌ detection‌ ‌for‌ ‌movies‌ ‌and‌ ‌group‌ ‌shots‌ ‌afterward‌ ‌to‌ ‌form‌ ‌scenes,‌ ‌where‌ ‌the‌ ‌scene‌
‌boundary‌ ‌ detection‌ ‌could‌ ‌be‌ ‌regarded‌ ‌as‌ ‌a‌ ‌binary‌ ‌classification‌ ‌problem‌ ‌on‌ ‌shot‌ ‌boundaries.‌ ‌
‌ ‌ In‌ ‌this‌ ‌challenge,‌ ‌your‌ ‌task‌ ‌is‌ ‌to‌ ‌predict‌ ‌the‌ ‌scene‌ ‌segmentation‌ ‌for‌ ‌each‌ ‌movie‌ ‌given‌
‌features‌ ‌ for‌ ‌each‌ ‌shot,‌ ‌and‌ ‌preliminary‌ ‌scene‌ ‌transition‌ ‌predictions.‌ ‌

**Data‌** ‌

We‌ ‌provide‌ ‌a‌ ‌dataset‌ ‌‌https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view‌,‌ ‌including‌ ‌64‌
‌movies‌ ‌with‌ ‌features‌ ‌of‌ ‌place,‌ ‌cast,‌ ‌action,‌ ‌and‌ ‌audio.‌ ‌ These‌ ‌features‌ ‌are‌ ‌encoded‌ ‌as‌
‌one-dimensional‌ ‌vectors,‌ ‌with‌ ‌dimensions‌ ‌of‌ ‌2048,‌ ‌512,‌ ‌512,‌ ‌ and‌ ‌512,‌ ‌respectively.‌ ‌Details‌ ‌of‌
‌feature‌ ‌extraction‌ ‌models‌ ‌can‌ ‌be‌ ‌found‌ ‌in‌ ‌this‌ ‌paper‌ ‌(Rao‌ ‌et‌ ‌ al.)
‌ https://arxiv.org/pdf/2004.02678.pdf‌.‌ ‌The‌ ‌first‌ ‌dimensions‌ ‌of‌ ‌these‌ ‌scene‌ ‌features‌ ‌indicate‌ ‌the‌
‌number‌ ‌of‌ ‌shots‌ ‌within‌ ‌that‌ ‌ scene.‌ ‌We‌ ‌also‌ ‌provide‌ ‌scene‌ ‌transition‌ ‌boundaries,‌ ‌both‌ ‌the‌
‌ground‌ ‌truth‌ ‌and‌ ‌preliminary‌ ‌ predictions.‌ ‌The‌ ‌'shot_end_frame'‌ ‌is‌ ‌the‌ ‌end‌ ‌frame‌ ‌index‌ ‌for‌
‌each‌ ‌shot,‌ ‌which‌ ‌is‌ ‌provided‌ ‌for‌ ‌ evaluation‌ ‌only.‌ ‌ Evaluation‌ ‌

We‌ ‌take‌ ‌two‌ ‌commonly‌ ‌used‌ ‌metrics:‌ ‌ 1.mAP‌ ‌--‌ ‌the‌ ‌mean‌ ‌Average‌ ‌Precision‌ ‌of‌ ‌scene‌ ‌transition‌
‌predictions‌ ‌for‌ ‌each‌ ‌movie.‌ ‌ 2.Miou‌ ‌--‌ ‌for‌ ‌each‌ ‌ground-truth‌ ‌scene,‌ ‌we‌ ‌take‌ ‌the‌ ‌maximum‌
‌intersection-over-union‌ ‌with‌ ‌ the‌ ‌detected‌ ‌scenes,‌ ‌averaging‌ ‌them‌ ‌on‌ ‌the‌ ‌whole‌ ‌video.‌ ‌Then‌ ‌the‌
‌same‌ ‌is‌ ‌done‌ ‌for‌ ‌ detected‌ ‌scenes‌ ‌against‌ ‌ground-truth‌ ‌scenes,‌ ‌and‌ ‌the‌ ‌two‌ ‌quantities‌ ‌are‌
‌again‌ ‌ averaged.‌ ‌The‌ ‌intersection/union‌ ‌is‌ ‌evaluated‌ ‌by‌ ‌counting‌ ‌the‌ ‌frames.‌ ‌ ‌ Implementation‌
‌of‌ ‌these‌ ‌metrics‌ ‌is‌ ‌available‌ ‌at‌ ‌our‌ ‌team’s‌ ‌github‌ ‌repo‌
‌‌https://github.com/eluv-io/elv-ml-challenge‌.‌ ‌

**_Instruction on How to Run_**

To evaluate the predicted accuracues with respect to the provided Ground Truths, place <imdb id>.pkl for all movies in
the dataset under data_dir, and run

`python main.py data_dir`