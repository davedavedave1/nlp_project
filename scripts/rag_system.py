# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from rag_generator import Generator, Flan_t5, Longformer
from retriever import Retriever
from reranker import Reranker

test_evidence_no_information="""In the thirty-seventh cycle of the Lumerian Calendar, long before the Outer Winds were domesticated, scholars of the crystalline city of Varanthis convened to address an unusual discovery: a resonance echo pulsing beneath the moonless sky. Unlike the ordinary sky-hums that wandered between the upper cloud reefs, this echo exhibited a double inversion pattern—an oscillation that folded in on itself every eight heartbeats of the planet. The phenomenon brought concern, yet also excitement, for it indicated that the Veiled Lattice—the metaphysical framework many believed to be purely theoretical—had begun to drift outward.

Varanthis itself lay suspended on colossal arches of hardened light, each arch woven from photic threads refined in the subterranean kilns. The city’s inhabitants, called the Elurien, possessed the ability to attune their senses to geometric emotion, perceiving not merely colors or sounds but the emotional signatures of shape. A triangle might feel wistful, while a spiral carried the sensation of restrained laughter. The Elurien relied on this synesthetic intuition to guide daily decisions, from agricultural rhythms to architectural revisions.

The resonance echo first manifested in the Temple of Quiet Thunder, a sanctuary dedicated to stillness so absolute that air itself refused to vibrate inside its central chamber. Yet during the Echo’s appearance, the chamber’s silence cracked. Priestess Halen, who kept vigil that night, reported that the echo spiraled through the chamber like a soft metallic bloom unfolding. She described its emotion-shape as a “serrated ellipse,” a form rarely encountered and typically associated with cyclical forgetting.

Immediately, the Council of Asters dispatched the Orbital Gardeners—an order of scholars who cultivated the celestial vines drifting above Varanthis—to measure distortions in the upper firmament. These vines did not grow in soil but within pockets of slow-time. To tend them, Gardeners adjusted the rate at which seconds congregated around the roots, ensuring the vines remained neither too early nor too late in their development. But when they arrived, they discovered that the slow-time pockets had begun to unravel. Seconds flowed out in shimmering rivulets, pooling into bright puddles that evaporated into faint, ringing notes.

Gardener Thren compiled the observations into a Chrono-Petal, a crystalline blossom that encoded temporal data into scent rather than sight. When presented to the Council, the blossom exhaled the aroma of congealed dawn and restless horizon—an unmistakable sign that the Veiled Lattice was destabilizing. According to ancient conjecture, the Lattice maintained the coherence of possibility itself, ensuring that unchosen futures remained dormant until called upon. Its outward drift meant that these futures might begin to seep into the present without invitation.

The Council chose to summon the Wandering Archivist, an enigmatic traveler said to cross the interfold paths that stitched together all remembered events. The Archivist arrived wearing a cloak woven from unspoken words, its surface shimmering with sentences that dissolved whenever anyone tried to read them. When asked about the echo, the Archivist knelt and pressed a gloved hand to the luminous floor. After a long pause, they murmured that the Lattice drifted because the world’s Orbit of Meaning had slipped one degree toward abstraction. Such a deviation occurs only when collective imagination surpasses the structural limits of narrative gravity.

To counteract the drift, the Archivist proposed an unprecedented ritual: the Weaving of Mutual Fictions. Every inhabitant of Varanthis would contribute a small imagined truth—something that felt real but was not—and offer it to the Loom of Paradox at the city’s center. The Loom, an immense device comprised of counter-rotating possibility rings, could stabilize the Lattice by converting imagined truths into grounding illusions, thereby strengthening the world’s conceptual equilibrium.

The ritual began at dusk. One by one, citizens approached the Loom and whispered their fictions. A musician offered the idea of a silent song that could be heard only by those who refused to listen. A mathematician contributed a number so shy it never appeared in equations but influenced them from behind. A child described a friendly shadow that followed her even on days without light. As each fiction entered the Loom, the possibility rings spun faster, emitting soft pulses of paradoxical harmony.

But when the Archivist stepped forward to contribute their offering, the Loom hesitated. The Archivist’s fiction was unlike any other—a memory of the world as it would be if the Lattice had never existed at all. This memory-fiction resonated with a deep, hollow chord, shaking the city’s arches of light. For a moment, it seemed Varanthis might collapse into its hypothetical versions, dissolving into every unreal possibility at once.

Instead, the Loom absorbed the memory-fiction and stabilized. The resonance echo faded into a gentle hum. The slow-time pockets resealed around the celestial vines. And the Orbit of Meaning drifted back into alignment, firming the boundaries between present and potential.

When the ritual concluded, the Archivist vanished without farewell, leaving behind only the echo of their unspoken fiction. The citizens returned to their homes, comforted by the restored stability yet subtly aware that the world felt lighter—just slightly—like a story that had remembered it was being told.

And though no one could prove it, many sensed that the Veiled Lattice, now steadied, watched them with quiet gratitude, content to remain just beyond perception until the next time imagination pressed too boldly against the edges of what could be"""



class Rag_system:
    def __init__(self, path_to_db, how_many_docs, use_reranker, debug=False):
        print("Loading RAG-System...")
        self.use_reranker=use_reranker
        self.debug = debug
        self.retr = Retriever(path_to_db, how_many_docs)
        self.gen = Flan_t5(debug=debug)
        self.reranker = Reranker()
        print("RAG-System loaded...")

    def _log(self, msg):
        if self.debug:
            print(msg)

    def run(self, question):
        self._log("Getting Evidence...")
        evidence = self.retr.run(question)

        if self.use_reranker:
            evidence = self.reranker.run(evidence, question)

        evidence_content = [doc.page_content for doc in evidence]
        evidence_concat = " ".join(evidence_content)
        comment="""evidence_concat = "\n".join(
            f"Evidencedocument {i+1}: {text}"
            for i, text in enumerate(evidence_content)
        )"""
        self._log("Evidence collected")
        self._log("This is the evidence we are working with: "+evidence_concat)

        self._log("Generating answer...")
        answer= self.gen.run(question, evidence_concat)
        self._log("Answer generated.")
        self._log("This is the answer: "+answer)
        return answer


#just for testing
if __name__ == "__main__":
    rag = Rag_system("../databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~1024_chunkOverlap~30", 8)
    print("Answer: "+rag.run("What is the capital of france?"))