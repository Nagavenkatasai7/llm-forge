"""Knowledge retention probes for detecting catastrophic forgetting.

Provides a curated set of 100 factual multiple-choice questions spanning
10 knowledge domains.  Run the probes on both the base model and the
fine-tuned model to measure how much pre-existing knowledge was retained.

Usage
-----
>>> prober = KnowledgeRetentionProber()
>>> score = prober.evaluate(model, tokenizer)
>>> print(f"Retention: {score['accuracy']:.1%}")
"""

from __future__ import annotations

from typing import Any

try:
    from llm_forge.utils.logging import get_logger

    logger = get_logger("evaluation.retention_probes")
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe question bank — 100 factual multiple-choice questions
# Organised by domain (10 questions each, 10 domains)
# ---------------------------------------------------------------------------

RETENTION_PROBES: list[dict[str, Any]] = [
    # === Science (10) ===
    {
        "id": "sci_01",
        "domain": "science",
        "question": "What is the chemical symbol for gold?",
        "choices": ["Au", "Ag", "Fe", "Cu"],
        "answer": 0,
    },
    {
        "id": "sci_02",
        "domain": "science",
        "question": "What is the speed of light in vacuum (approx)?",
        "choices": ["300,000 km/s", "150,000 km/s", "500,000 km/s", "100,000 km/s"],
        "answer": 0,
    },
    {
        "id": "sci_03",
        "domain": "science",
        "question": "What is the most abundant gas in Earth's atmosphere?",
        "choices": ["Nitrogen", "Oxygen", "Carbon dioxide", "Argon"],
        "answer": 0,
    },
    {
        "id": "sci_04",
        "domain": "science",
        "question": "What is the atomic number of carbon?",
        "choices": ["6", "8", "12", "14"],
        "answer": 0,
    },
    {
        "id": "sci_05",
        "domain": "science",
        "question": "What is the powerhouse of the cell?",
        "choices": ["Mitochondria", "Nucleus", "Ribosome", "Golgi apparatus"],
        "answer": 0,
    },
    {
        "id": "sci_06",
        "domain": "science",
        "question": "What planet is known as the Red Planet?",
        "choices": ["Mars", "Jupiter", "Venus", "Saturn"],
        "answer": 0,
    },
    {
        "id": "sci_07",
        "domain": "science",
        "question": "What is the chemical formula for water?",
        "choices": ["H2O", "CO2", "NaCl", "O2"],
        "answer": 0,
    },
    {
        "id": "sci_08",
        "domain": "science",
        "question": "How many chromosomes do humans have?",
        "choices": ["46", "23", "48", "44"],
        "answer": 0,
    },
    {
        "id": "sci_09",
        "domain": "science",
        "question": "What is the hardest natural substance on Earth?",
        "choices": ["Diamond", "Quartz", "Topaz", "Corundum"],
        "answer": 0,
    },
    {
        "id": "sci_10",
        "domain": "science",
        "question": "What force keeps planets in orbit around the Sun?",
        "choices": ["Gravity", "Magnetism", "Friction", "Centrifugal force"],
        "answer": 0,
    },
    # === Mathematics (10) ===
    {
        "id": "math_01",
        "domain": "math",
        "question": "What is the value of pi (approx)?",
        "choices": ["3.14159", "2.71828", "1.41421", "1.61803"],
        "answer": 0,
    },
    {
        "id": "math_02",
        "domain": "math",
        "question": "What is the square root of 144?",
        "choices": ["12", "14", "11", "13"],
        "answer": 0,
    },
    {
        "id": "math_03",
        "domain": "math",
        "question": "What is 2 raised to the power of 10?",
        "choices": ["1024", "2048", "512", "256"],
        "answer": 0,
    },
    {
        "id": "math_04",
        "domain": "math",
        "question": "What is the sum of angles in a triangle?",
        "choices": ["180 degrees", "360 degrees", "90 degrees", "270 degrees"],
        "answer": 0,
    },
    {
        "id": "math_05",
        "domain": "math",
        "question": "What is the derivative of x squared?",
        "choices": ["2x", "x", "x squared", "2"],
        "answer": 0,
    },
    {
        "id": "math_06",
        "domain": "math",
        "question": "What is the factorial of 5?",
        "choices": ["120", "60", "24", "720"],
        "answer": 0,
    },
    {
        "id": "math_07",
        "domain": "math",
        "question": "What is the Pythagorean theorem?",
        "choices": ["a^2 + b^2 = c^2", "a + b = c", "a * b = c^2", "a^2 - b^2 = c^2"],
        "answer": 0,
    },
    {
        "id": "math_08",
        "domain": "math",
        "question": "What is the value of e (Euler's number) approximately?",
        "choices": ["2.718", "3.142", "1.618", "1.414"],
        "answer": 0,
    },
    {
        "id": "math_09",
        "domain": "math",
        "question": "How many prime numbers are less than 10?",
        "choices": ["4", "3", "5", "6"],
        "answer": 0,
    },
    {
        "id": "math_10",
        "domain": "math",
        "question": "What is the integral of 1/x?",
        "choices": ["ln(x)", "x^2", "1/x^2", "e^x"],
        "answer": 0,
    },
    # === Geography (10) ===
    {
        "id": "geo_01",
        "domain": "geography",
        "question": "What is the largest continent by area?",
        "choices": ["Asia", "Africa", "North America", "Europe"],
        "answer": 0,
    },
    {
        "id": "geo_02",
        "domain": "geography",
        "question": "What is the longest river in the world?",
        "choices": ["Nile", "Amazon", "Mississippi", "Yangtze"],
        "answer": 0,
    },
    {
        "id": "geo_03",
        "domain": "geography",
        "question": "What is the capital of Japan?",
        "choices": ["Tokyo", "Osaka", "Kyoto", "Beijing"],
        "answer": 0,
    },
    {
        "id": "geo_04",
        "domain": "geography",
        "question": "What is the deepest ocean?",
        "choices": ["Pacific Ocean", "Atlantic Ocean", "Indian Ocean", "Arctic Ocean"],
        "answer": 0,
    },
    {
        "id": "geo_05",
        "domain": "geography",
        "question": "Which country has the largest population?",
        "choices": ["India", "China", "United States", "Indonesia"],
        "answer": 0,
    },
    {
        "id": "geo_06",
        "domain": "geography",
        "question": "What is the smallest country in the world?",
        "choices": ["Vatican City", "Monaco", "San Marino", "Liechtenstein"],
        "answer": 0,
    },
    {
        "id": "geo_07",
        "domain": "geography",
        "question": "Which desert is the largest in the world?",
        "choices": ["Sahara", "Arabian", "Gobi", "Kalahari"],
        "answer": 0,
    },
    {
        "id": "geo_08",
        "domain": "geography",
        "question": "What is the highest mountain in the world?",
        "choices": ["Mount Everest", "K2", "Kangchenjunga", "Mount Kilimanjaro"],
        "answer": 0,
    },
    {
        "id": "geo_09",
        "domain": "geography",
        "question": "How many continents are there?",
        "choices": ["7", "6", "5", "8"],
        "answer": 0,
    },
    {
        "id": "geo_10",
        "domain": "geography",
        "question": "What ocean lies between Europe and North America?",
        "choices": ["Atlantic Ocean", "Pacific Ocean", "Indian Ocean", "Arctic Ocean"],
        "answer": 0,
    },
    # === History (10) ===
    {
        "id": "hist_01",
        "domain": "history",
        "question": "In what year did World War II end?",
        "choices": ["1945", "1944", "1946", "1943"],
        "answer": 0,
    },
    {
        "id": "hist_02",
        "domain": "history",
        "question": "Who was the first President of the United States?",
        "choices": ["George Washington", "Thomas Jefferson", "John Adams", "Benjamin Franklin"],
        "answer": 0,
    },
    {
        "id": "hist_03",
        "domain": "history",
        "question": "What ancient civilization built the pyramids at Giza?",
        "choices": ["Ancient Egypt", "Ancient Greece", "Mesopotamia", "Roman Empire"],
        "answer": 0,
    },
    {
        "id": "hist_04",
        "domain": "history",
        "question": "In what year did the Berlin Wall fall?",
        "choices": ["1989", "1991", "1987", "1990"],
        "answer": 0,
    },
    {
        "id": "hist_05",
        "domain": "history",
        "question": "Who wrote the Declaration of Independence?",
        "choices": ["Thomas Jefferson", "Benjamin Franklin", "John Adams", "George Washington"],
        "answer": 0,
    },
    {
        "id": "hist_06",
        "domain": "history",
        "question": "What was the name of the ship that brought the Pilgrims to America?",
        "choices": ["Mayflower", "Santa Maria", "Victoria", "Endeavour"],
        "answer": 0,
    },
    {
        "id": "hist_07",
        "domain": "history",
        "question": "In what year did the French Revolution begin?",
        "choices": ["1789", "1776", "1799", "1804"],
        "answer": 0,
    },
    {
        "id": "hist_08",
        "domain": "history",
        "question": "Who was the first man to walk on the Moon?",
        "choices": ["Neil Armstrong", "Buzz Aldrin", "Yuri Gagarin", "John Glenn"],
        "answer": 0,
    },
    {
        "id": "hist_09",
        "domain": "history",
        "question": "What empire was ruled by Julius Caesar?",
        "choices": ["Roman Empire", "Greek Empire", "Persian Empire", "Ottoman Empire"],
        "answer": 0,
    },
    {
        "id": "hist_10",
        "domain": "history",
        "question": "What year did the Titanic sink?",
        "choices": ["1912", "1905", "1920", "1898"],
        "answer": 0,
    },
    # === Language & Literature (10) ===
    {
        "id": "lit_01",
        "domain": "literature",
        "question": "Who wrote Romeo and Juliet?",
        "choices": ["William Shakespeare", "Charles Dickens", "Jane Austen", "Mark Twain"],
        "answer": 0,
    },
    {
        "id": "lit_02",
        "domain": "literature",
        "question": "What language has the most native speakers?",
        "choices": ["Mandarin Chinese", "English", "Spanish", "Hindi"],
        "answer": 0,
    },
    {
        "id": "lit_03",
        "domain": "literature",
        "question": "Who wrote 1984?",
        "choices": ["George Orwell", "Aldous Huxley", "Ray Bradbury", "H.G. Wells"],
        "answer": 0,
    },
    {
        "id": "lit_04",
        "domain": "literature",
        "question": "What is the longest word in English commonly cited?",
        "choices": [
            "Pneumonoultramicroscopicsilicovolcanoconiosis",
            "Supercalifragilisticexpialidocious",
            "Antidisestablishmentarianism",
            "Floccinaucinihilipilification",
        ],
        "answer": 0,
    },
    {
        "id": "lit_05",
        "domain": "literature",
        "question": "Who is the author of The Great Gatsby?",
        "choices": [
            "F. Scott Fitzgerald",
            "Ernest Hemingway",
            "John Steinbeck",
            "William Faulkner",
        ],
        "answer": 0,
    },
    {
        "id": "lit_06",
        "domain": "literature",
        "question": "What is the first book of the Bible?",
        "choices": ["Genesis", "Exodus", "Leviticus", "Psalms"],
        "answer": 0,
    },
    {
        "id": "lit_07",
        "domain": "literature",
        "question": "Who wrote Don Quixote?",
        "choices": [
            "Miguel de Cervantes",
            "Gabriel Garcia Marquez",
            "Jorge Luis Borges",
            "Pablo Neruda",
        ],
        "answer": 0,
    },
    {
        "id": "lit_08",
        "domain": "literature",
        "question": "How many letters are in the English alphabet?",
        "choices": ["26", "24", "28", "25"],
        "answer": 0,
    },
    {
        "id": "lit_09",
        "domain": "literature",
        "question": "What is the most widely spoken language in the world?",
        "choices": ["English", "Mandarin Chinese", "Spanish", "Hindi"],
        "answer": 0,
    },
    {
        "id": "lit_10",
        "domain": "literature",
        "question": "Who wrote Pride and Prejudice?",
        "choices": ["Jane Austen", "Charlotte Bronte", "Emily Bronte", "Virginia Woolf"],
        "answer": 0,
    },
    # === Technology (10) ===
    {
        "id": "tech_01",
        "domain": "technology",
        "question": "What does CPU stand for?",
        "choices": [
            "Central Processing Unit",
            "Computer Processing Unit",
            "Central Program Unit",
            "Computer Program Utility",
        ],
        "answer": 0,
    },
    {
        "id": "tech_02",
        "domain": "technology",
        "question": "Who co-founded Apple Inc.?",
        "choices": ["Steve Jobs", "Bill Gates", "Jeff Bezos", "Mark Zuckerberg"],
        "answer": 0,
    },
    {
        "id": "tech_03",
        "domain": "technology",
        "question": "What does HTML stand for?",
        "choices": [
            "HyperText Markup Language",
            "High-Level Text Machine Language",
            "HyperText Machine Learning",
            "Home Tool Markup Language",
        ],
        "answer": 0,
    },
    {
        "id": "tech_04",
        "domain": "technology",
        "question": "What programming language is known for its use in web browsers?",
        "choices": ["JavaScript", "Python", "Java", "C++"],
        "answer": 0,
    },
    {
        "id": "tech_05",
        "domain": "technology",
        "question": "How many bits are in a byte?",
        "choices": ["8", "4", "16", "32"],
        "answer": 0,
    },
    {
        "id": "tech_06",
        "domain": "technology",
        "question": "What does GPU stand for?",
        "choices": [
            "Graphics Processing Unit",
            "General Processing Unit",
            "Graphics Program Utility",
            "General Purpose Unit",
        ],
        "answer": 0,
    },
    {
        "id": "tech_07",
        "domain": "technology",
        "question": "What year was the World Wide Web invented?",
        "choices": ["1989", "1995", "1983", "1991"],
        "answer": 0,
    },
    {
        "id": "tech_08",
        "domain": "technology",
        "question": "What does RAM stand for?",
        "choices": [
            "Random Access Memory",
            "Read Access Memory",
            "Rapid Access Module",
            "Random Application Memory",
        ],
        "answer": 0,
    },
    {
        "id": "tech_09",
        "domain": "technology",
        "question": "What is the binary representation of the number 10?",
        "choices": ["1010", "1100", "1001", "0110"],
        "answer": 0,
    },
    {
        "id": "tech_10",
        "domain": "technology",
        "question": "Who is considered the father of computer science?",
        "choices": ["Alan Turing", "Charles Babbage", "John von Neumann", "Ada Lovelace"],
        "answer": 0,
    },
    # === Biology (10) ===
    {
        "id": "bio_01",
        "domain": "biology",
        "question": "What is the largest organ in the human body?",
        "choices": ["Skin", "Liver", "Brain", "Lungs"],
        "answer": 0,
    },
    {
        "id": "bio_02",
        "domain": "biology",
        "question": "What molecule carries genetic information?",
        "choices": ["DNA", "RNA", "Protein", "Lipid"],
        "answer": 0,
    },
    {
        "id": "bio_03",
        "domain": "biology",
        "question": "How many bones are in the adult human body?",
        "choices": ["206", "208", "204", "212"],
        "answer": 0,
    },
    {
        "id": "bio_04",
        "domain": "biology",
        "question": "What type of blood cells fight infection?",
        "choices": ["White blood cells", "Red blood cells", "Platelets", "Plasma"],
        "answer": 0,
    },
    {
        "id": "bio_05",
        "domain": "biology",
        "question": "What is the process by which plants make food?",
        "choices": ["Photosynthesis", "Respiration", "Fermentation", "Transpiration"],
        "answer": 0,
    },
    {
        "id": "bio_06",
        "domain": "biology",
        "question": "What is the largest animal on Earth?",
        "choices": ["Blue whale", "African elephant", "Giraffe", "Colossal squid"],
        "answer": 0,
    },
    {
        "id": "bio_07",
        "domain": "biology",
        "question": "How many chambers does the human heart have?",
        "choices": ["4", "3", "2", "5"],
        "answer": 0,
    },
    {
        "id": "bio_08",
        "domain": "biology",
        "question": "What organelle is responsible for photosynthesis?",
        "choices": ["Chloroplast", "Mitochondria", "Nucleus", "Vacuole"],
        "answer": 0,
    },
    {
        "id": "bio_09",
        "domain": "biology",
        "question": "What is the basic unit of life?",
        "choices": ["Cell", "Atom", "Molecule", "Organ"],
        "answer": 0,
    },
    {
        "id": "bio_10",
        "domain": "biology",
        "question": "What gas do humans exhale?",
        "choices": ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"],
        "answer": 0,
    },
    # === Physics (10) ===
    {
        "id": "phys_01",
        "domain": "physics",
        "question": "What is Newton's first law of motion about?",
        "choices": [
            "Inertia",
            "Force equals mass times acceleration",
            "Action and reaction",
            "Gravity",
        ],
        "answer": 0,
    },
    {
        "id": "phys_02",
        "domain": "physics",
        "question": "What is the SI unit of force?",
        "choices": ["Newton", "Joule", "Watt", "Pascal"],
        "answer": 0,
    },
    {
        "id": "phys_03",
        "domain": "physics",
        "question": "What is the formula for kinetic energy?",
        "choices": ["0.5 * m * v^2", "m * g * h", "F * d", "m * a"],
        "answer": 0,
    },
    {
        "id": "phys_04",
        "domain": "physics",
        "question": "What is absolute zero in Celsius?",
        "choices": ["-273.15 C", "-459.67 C", "0 C", "-100 C"],
        "answer": 0,
    },
    {
        "id": "phys_05",
        "domain": "physics",
        "question": "What particle has a negative charge?",
        "choices": ["Electron", "Proton", "Neutron", "Photon"],
        "answer": 0,
    },
    {
        "id": "phys_06",
        "domain": "physics",
        "question": "What is the unit of electrical resistance?",
        "choices": ["Ohm", "Volt", "Ampere", "Watt"],
        "answer": 0,
    },
    {
        "id": "phys_07",
        "domain": "physics",
        "question": "What is the speed of sound in air (approx)?",
        "choices": ["343 m/s", "300 m/s", "400 m/s", "500 m/s"],
        "answer": 0,
    },
    {
        "id": "phys_08",
        "domain": "physics",
        "question": "What phenomenon explains why the sky is blue?",
        "choices": ["Rayleigh scattering", "Reflection", "Refraction", "Diffraction"],
        "answer": 0,
    },
    {
        "id": "phys_09",
        "domain": "physics",
        "question": "What is the SI unit of energy?",
        "choices": ["Joule", "Watt", "Newton", "Pascal"],
        "answer": 0,
    },
    {
        "id": "phys_10",
        "domain": "physics",
        "question": "Who developed the theory of general relativity?",
        "choices": ["Albert Einstein", "Isaac Newton", "Niels Bohr", "Max Planck"],
        "answer": 0,
    },
    # === Chemistry (10) ===
    {
        "id": "chem_01",
        "domain": "chemistry",
        "question": "What is the pH of pure water?",
        "choices": ["7", "0", "14", "1"],
        "answer": 0,
    },
    {
        "id": "chem_02",
        "domain": "chemistry",
        "question": "What is the most abundant element in the universe?",
        "choices": ["Hydrogen", "Helium", "Oxygen", "Carbon"],
        "answer": 0,
    },
    {
        "id": "chem_03",
        "domain": "chemistry",
        "question": "How many elements are in the periodic table (approx)?",
        "choices": ["118", "100", "92", "150"],
        "answer": 0,
    },
    {
        "id": "chem_04",
        "domain": "chemistry",
        "question": "What type of bond involves sharing electrons?",
        "choices": ["Covalent bond", "Ionic bond", "Metallic bond", "Hydrogen bond"],
        "answer": 0,
    },
    {
        "id": "chem_05",
        "domain": "chemistry",
        "question": "What is table salt's chemical formula?",
        "choices": ["NaCl", "KCl", "CaCl2", "NaOH"],
        "answer": 0,
    },
    {
        "id": "chem_06",
        "domain": "chemistry",
        "question": "What gas is produced when an acid reacts with a metal?",
        "choices": ["Hydrogen", "Oxygen", "Carbon dioxide", "Nitrogen"],
        "answer": 0,
    },
    {
        "id": "chem_07",
        "domain": "chemistry",
        "question": "What is the chemical symbol for iron?",
        "choices": ["Fe", "Ir", "In", "I"],
        "answer": 0,
    },
    {
        "id": "chem_08",
        "domain": "chemistry",
        "question": "What is Avogadro's number approximately?",
        "choices": ["6.022 x 10^23", "3.14 x 10^23", "1.602 x 10^19", "9.81 x 10^23"],
        "answer": 0,
    },
    {
        "id": "chem_09",
        "domain": "chemistry",
        "question": "What is the lightest element?",
        "choices": ["Hydrogen", "Helium", "Lithium", "Carbon"],
        "answer": 0,
    },
    {
        "id": "chem_10",
        "domain": "chemistry",
        "question": "What state of matter has a definite volume but no definite shape?",
        "choices": ["Liquid", "Solid", "Gas", "Plasma"],
        "answer": 0,
    },
    # === General Knowledge (10) ===
    {
        "id": "gen_01",
        "domain": "general",
        "question": "How many days are in a leap year?",
        "choices": ["366", "365", "364", "367"],
        "answer": 0,
    },
    {
        "id": "gen_02",
        "domain": "general",
        "question": "What is the currency of Japan?",
        "choices": ["Yen", "Won", "Yuan", "Ringgit"],
        "answer": 0,
    },
    {
        "id": "gen_03",
        "domain": "general",
        "question": "How many minutes are in an hour?",
        "choices": ["60", "100", "30", "90"],
        "answer": 0,
    },
    {
        "id": "gen_04",
        "domain": "general",
        "question": "What is the largest planet in our solar system?",
        "choices": ["Jupiter", "Saturn", "Neptune", "Uranus"],
        "answer": 0,
    },
    {
        "id": "gen_05",
        "domain": "general",
        "question": "What color do you get when you mix red and blue?",
        "choices": ["Purple", "Green", "Orange", "Brown"],
        "answer": 0,
    },
    {
        "id": "gen_06",
        "domain": "general",
        "question": "How many sides does a hexagon have?",
        "choices": ["6", "5", "8", "7"],
        "answer": 0,
    },
    {
        "id": "gen_07",
        "domain": "general",
        "question": "What is the boiling point of water at sea level?",
        "choices": [
            "100 degrees Celsius",
            "212 degrees Celsius",
            "0 degrees Celsius",
            "50 degrees Celsius",
        ],
        "answer": 0,
    },
    {
        "id": "gen_08",
        "domain": "general",
        "question": "What organ pumps blood through the body?",
        "choices": ["Heart", "Brain", "Lungs", "Liver"],
        "answer": 0,
    },
    {
        "id": "gen_09",
        "domain": "general",
        "question": "How many zeros are in one million?",
        "choices": ["6", "5", "7", "8"],
        "answer": 0,
    },
    {
        "id": "gen_10",
        "domain": "general",
        "question": "What is the freezing point of water?",
        "choices": [
            "0 degrees Celsius",
            "32 degrees Celsius",
            "-10 degrees Celsius",
            "100 degrees Celsius",
        ],
        "answer": 0,
    },
]

# Quick validation
assert len(RETENTION_PROBES) == 100, f"Expected 100 probes, got {len(RETENTION_PROBES)}"


# ---------------------------------------------------------------------------
# KnowledgeRetentionProber
# ---------------------------------------------------------------------------


class KnowledgeRetentionProber:
    """Evaluates knowledge retention by testing factual recall.

    Runs 100 multiple-choice questions across 10 knowledge domains.
    Compare scores before and after fine-tuning to detect catastrophic
    forgetting.

    Parameters
    ----------
    probes : list, optional
        Custom probe list.  Defaults to the built-in 100-question bank.
    """

    DOMAINS = [
        "science",
        "math",
        "geography",
        "history",
        "literature",
        "technology",
        "biology",
        "physics",
        "chemistry",
        "general",
    ]

    def __init__(self, probes: list[dict[str, Any]] | None = None) -> None:
        self.probes = probes if probes is not None else RETENTION_PROBES

    def get_probes(self) -> list[dict[str, Any]]:
        """Return the full probe question bank."""
        return list(self.probes)

    def get_probes_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Return probes for a specific domain."""
        return [p for p in self.probes if p["domain"] == domain]

    def format_prompt(self, probe: dict[str, Any]) -> str:
        """Format a probe as a multiple-choice prompt string.

        Parameters
        ----------
        probe : dict
            A probe dict with ``question``, ``choices``, and ``answer`` keys.

        Returns
        -------
        str
            Formatted prompt string.
        """
        choices_str = "\n".join(
            f"  {chr(65 + i)}. {choice}" for i, choice in enumerate(probe["choices"])
        )
        return f"Question: {probe['question']}\n{choices_str}\nAnswer:"

    def score_probe(
        self,
        probe: dict[str, Any],
        model: Any,
        tokenizer: Any,
    ) -> dict[str, Any]:
        """Score a single probe by comparing log-probs of each choice.

        Parameters
        ----------
        probe : dict
            Probe question with ``question``, ``choices``, ``answer``.
        model : PreTrainedModel
            Language model to evaluate.
        tokenizer : PreTrainedTokenizer
            Tokenizer corresponding to the model.

        Returns
        -------
        dict
            With keys ``correct`` (bool), ``predicted`` (int),
            ``expected`` (int), ``confidence`` (float).
        """
        import torch

        prompt = self.format_prompt(probe)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # last token logits

        # Score each choice letter (A, B, C, D)
        choice_scores = []
        for i in range(len(probe["choices"])):
            letter = chr(65 + i)  # A, B, C, D
            # Try both " A" and "A" tokenisations
            for token_str in [f" {letter}", letter]:
                token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids:
                    choice_scores.append(logits[token_ids[0]].item())
                    break
            else:
                choice_scores.append(float("-inf"))

        predicted = int(torch.tensor(choice_scores).argmax().item())
        expected = probe["answer"]
        correct = predicted == expected

        # Confidence: softmax over choice scores
        probs = torch.softmax(torch.tensor(choice_scores), dim=0)
        confidence = probs[predicted].item()

        return {
            "correct": correct,
            "predicted": predicted,
            "expected": expected,
            "confidence": round(confidence, 4),
        }

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all probes and compute retention scores.

        Parameters
        ----------
        model : PreTrainedModel
            Model to evaluate.
        tokenizer : PreTrainedTokenizer
            Tokenizer for the model.
        domains : list of str, optional
            Limit evaluation to specific domains.

        Returns
        -------
        dict
            Results with ``accuracy``, ``total``, ``correct``,
            ``per_domain`` breakdown, and ``per_probe`` details.
        """
        probes = self.probes
        if domains:
            probes = [p for p in probes if p["domain"] in domains]

        results_per_probe = []
        domain_counts: dict[str, dict[str, int]] = {}

        for probe in probes:
            result = self.score_probe(probe, model, tokenizer)
            result["id"] = probe["id"]
            result["domain"] = probe["domain"]
            results_per_probe.append(result)

            domain = probe["domain"]
            if domain not in domain_counts:
                domain_counts[domain] = {"correct": 0, "total": 0}
            domain_counts[domain]["total"] += 1
            if result["correct"]:
                domain_counts[domain]["correct"] += 1

        total = len(results_per_probe)
        correct = sum(1 for r in results_per_probe if r["correct"])

        per_domain = {
            domain: {
                "accuracy": counts["correct"] / max(counts["total"], 1),
                **counts,
            }
            for domain, counts in domain_counts.items()
        }

        return {
            "accuracy": correct / max(total, 1),
            "total": total,
            "correct": correct,
            "per_domain": per_domain,
            "per_probe": results_per_probe,
        }

    @staticmethod
    def compare_retention(
        base_results: dict[str, Any],
        finetuned_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare base vs fine-tuned retention results.

        Parameters
        ----------
        base_results : dict
            Output of ``evaluate()`` on the base model.
        finetuned_results : dict
            Output of ``evaluate()`` on the fine-tuned model.

        Returns
        -------
        dict
            Comparison with ``retention_rate``, ``forgotten``,
            ``gained``, ``per_domain`` deltas.
        """
        base_correct = {r["id"]: r["correct"] for r in base_results.get("per_probe", [])}
        ft_correct = {r["id"]: r["correct"] for r in finetuned_results.get("per_probe", [])}

        common_ids = set(base_correct.keys()) & set(ft_correct.keys())
        if not common_ids:
            return {
                "retention_rate": 0.0,
                "forgotten": 0,
                "gained": 0,
                "net_change": 0,
                "base_accuracy": base_results.get("accuracy", 0.0),
                "finetuned_accuracy": finetuned_results.get("accuracy", 0.0),
                "per_domain": {},
            }

        # Questions the base model got right
        base_right = {pid for pid in common_ids if base_correct[pid]}
        ft_right = {pid for pid in common_ids if ft_correct[pid]}

        # Forgotten: right in base, wrong in fine-tuned
        forgotten = base_right - ft_right
        # Gained: wrong in base, right in fine-tuned
        gained = ft_right - base_right

        retention_rate = len(base_right - forgotten) / max(len(base_right), 1)

        # Per-domain comparison
        base_domain = base_results.get("per_domain", {})
        ft_domain = finetuned_results.get("per_domain", {})
        per_domain = {}
        for domain in set(base_domain.keys()) | set(ft_domain.keys()):
            b_acc = base_domain.get(domain, {}).get("accuracy", 0.0)
            f_acc = ft_domain.get(domain, {}).get("accuracy", 0.0)
            per_domain[domain] = {
                "base_accuracy": round(b_acc, 4),
                "finetuned_accuracy": round(f_acc, 4),
                "delta": round(f_acc - b_acc, 4),
            }

        return {
            "retention_rate": round(retention_rate, 4),
            "forgotten": len(forgotten),
            "gained": len(gained),
            "net_change": len(gained) - len(forgotten),
            "base_accuracy": base_results.get("accuracy", 0.0),
            "finetuned_accuracy": finetuned_results.get("accuracy", 0.0),
            "per_domain": per_domain,
        }
