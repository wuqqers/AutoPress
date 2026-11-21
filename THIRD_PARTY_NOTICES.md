# Speech embedding models

AutoPress uses the pretrained Google `speech_embedding/1` CNN, converted to ONNX
by David Scripka's openWakeWord project. The mel front end and speech embedding
are shipped from openWakeWord v0.5.1, without the pretrained wake-word classifiers.
AutoPress fits its own classification head locally from user-provided examples.

Sources:

- https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1
- https://github.com/dscripka/openWakeWord
- https://tfhub.dev/google/speech_embedding/1
- https://arxiv.org/abs/2002.01322

The openWakeWord README identifies the Google embedding backbone as Apache-2.0
licensed, and its implementation code is Apache-2.0 (Copyright 2022 David Scripka).
Its separate pretrained wake-word classifiers are described as CC BY-NC-SA 4.0;
those classifiers are not included in AutoPress. Refer to upstream model and
license documentation before redistributing modified model artifacts.

Apache-2.0 license text: src/autopress/resources/neural/LICENSE-Apache-2.0.txt
(resources/neural/LICENSE-Apache-2.0.txt inside the executable).

Artifact SHA-256 hashes are pinned and checked in `autopress.neural.HASHES`.
