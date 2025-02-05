# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "lean-interact",
#     "rich",
# ]
# ///
import json
import os
import sys

from datasets import load_dataset
from rich.syntax import Syntax
from tqdm import tqdm

from lean_interact import AutoLeanServer, LeanREPLConfig

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.insert(0, current_directory)

from utils import (
    clean_last_theorem_string,
    console,
    extract_exact_proof,
    indent_code,
    is_valid_lean,
    split_conclusion,
)


def beql(
    formalization_1: str, formalization_2: str, src_header: str, repl_config: LeanREPLConfig, timeout_per_proof: int
) -> bool:
    """
    Check if two formalizations are equivalent using the BEqL metric (https://arxiv.org/abs/2406.07222).
    """

    server = AutoLeanServer(config=repl_config)
    context_env = server.run_code(src_header, add_to_session_cache=True)["env"]

    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    res = [False, False]
    for i, (base_thm, reform_thm) in enumerate(
        [(formalization_1, formalization_2), (formalization_2, formalization_1)]
    ):
        try:
            formal_1_code = clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True) + "\n\n"
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} by"
        except ValueError:
            # we cannot change one of the theorems name, probably because it is invalid, so we skip
            continue

        def check_proof_sub(proof: str, formal_code: str = formal_1_code + formal_2_code) -> str | None:
            prepended_proof = "\nintros\nsymm_saturate\n"
            try:
                lean_output = server.run_code(
                    formal_code + indent_code(prepended_proof + proof, 2),
                    env=context_env,
                    timeout=timeout_per_proof,
                )

                if proof == "sorry":
                    # sorry is used on purpose to check if the formalization is well-typed
                    if is_valid_lean(lean_output, start_line=formal_2_start_line):
                        return proof
                    return None

                if is_valid_lean(lean_output, start_line=formal_2_start_line, allow_sorry=False):
                    if proof == "exact?":
                        return extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
                    return proof
            except TimeoutError:
                pass
            except (ConnectionAbortedError, json.JSONDecodeError) as e:
                console.log(f"Error during proof checking: {e}")
                pass

            return None

        # to avoid losing time, we first check if the formalizations are well-typed
        if check_proof_sub("sorry") is None:
            continue

        # 1. Use `exact?` tactic. We check if it is using the base theorem in the proof.
        proof_exact = check_proof_sub("exact?")
        if proof_exact and base_thm_name in proof_exact:
            res[i] = True
            continue

    return res[0] and res[1]


def beq_plus(
    formalization_1: str, formalization_2: str, src_header: str, repl_config: LeanREPLConfig, timeout_per_proof: int
) -> bool:
    """
    Check if two formalizations are equivalent using the BEq+ metric (https://arxiv.org/abs/2406.07222).
    """

    server = AutoLeanServer(config=repl_config)
    context_env = server.run_code(src_header, add_to_session_cache=True)["env"]

    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
        prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = [False, False]
    for i, (base_thm, reform_thm) in enumerate(
        [(formalization_1, formalization_2), (formalization_2, formalization_1)]
    ):
        try:
            formal_1_code = clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True) + "\n\n"
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} by"
        except ValueError:
            # we cannot change one of the theorems name, probably because it is invalid, so we skip
            continue

        def check_proof_sub(proof: str, formal_code: str = formal_1_code + formal_2_code) -> str | None:
            prepended_proof = "\nintros\nsymm_saturate\n"
            try:
                lean_output = server.run_code(
                    formal_code + indent_code(prepended_proof + proof, 2),
                    env=context_env,
                    timeout=timeout_per_proof,
                )

                if proof == "sorry":
                    # sorry is used on purpose to check if the formalization is well-typed
                    if is_valid_lean(lean_output, start_line=formal_2_start_line):
                        return proof
                    return None

                if is_valid_lean(lean_output, start_line=formal_2_start_line, allow_sorry=False):
                    if proof == "exact?":
                        return extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
                    return proof
            except TimeoutError:
                pass
            except (ConnectionAbortedError, json.JSONDecodeError) as e:
                console.log(f"Error during proof checking: {e}")
                pass

            return None

        # to avoid losing time, we first check if the formalizations are well-typed
        if check_proof_sub("sorry") is None:
            continue

        # 1. Use `exact?` tactic. We check if it is using the base theorem in the proof.
        proof_exact = check_proof_sub("exact?")
        if proof_exact and base_thm_name in proof_exact:
            res[i] = True
            continue

        # 2. try to apply the base theorem directly
        proof_apply = check_proof_sub(f"apply {base_thm_name}\n" + proof_all_apply)
        if proof_apply:
            res[i] = True
            continue

        # 3. try to add the conlusion of the base theorem as hypothesis
        # sanity check: if we can prove `reform_thm` using a tactic in `solver_tactics_have` without introducing the hypothesis,
        # then we should skip this `have` step as it may introduce a false positive
        # drawback of `have` strategy: variable names/types must match exactly
        provable_without_have = False
        try:
            provable_without_have = is_valid_lean(
                server.run_code(formal_2_code + proof_all_have, env=context_env, timeout=timeout_per_proof),
                allow_sorry=False,
            )
        except TimeoutError:
            pass
        except (ConnectionAbortedError, json.JSONDecodeError) as e:
            console.log(f"Error during proof checking: {e}")
            pass

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                    f"have {conclusion} := by\n"
                    + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                    + "\n"
                )
                proof_have = check_proof_sub(have_stmt_proof + proof_all_have)
                if proof_have:
                    res[i] = True
                    continue

        # 4. try to apply the base theorem with some tolerance on the differences in the conclusion
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply
            )
            if proof_convert:
                res[i] = True
                break

    return res[0] and res[1]


def examples_limitations(metric):
    repl_config = LeanREPLConfig(lean_version="v4.14.0", require="mathlib")

    src_header = """import Mathlib

open Fintype Group Monoid
open Set Real Ideal Polynomial
open scoped BigOperators"""

    formalization_pairs = [
        (
            "theorem random_name_1 {G : Type*} [Group G] [Fintype G] (h : Fintype.card G % 2 = 0) :\n  ∃ a : G, a ≠ 1 ∧ a = a⁻¹ :=",
            "theorem random_name_2 {G : Type*} [Group G] [Fintype G] (hG2 : Even (card G)) :\n  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=",
        ),
        (
            "theorem sPpp {G : Type*} [Group G] [Fintype G] {p q : ℕ} (hp : Prime p) (hq : Prime q) (hG : card G = p*q) :  IsSimpleGroup G → False :=",
            "theorem sQqq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (G : Type _) [Group G] [Fintype G] (hG : Fintype.card G = p * q) : ¬ IsSimpleGroup G :=",
        ),
        (
            "theorem sPppp {f : ℝ → ℝ} (hf : ∀ x y, |f x - f y| ≤ |x - y| ^ 2) : ∃ c, f = λ x => c :=",
            "theorem sQqqq (f : ℝ → ℝ) (h : ∀ (t x : ℝ), |f t - f x| ≤ |t - x| ^ 2) (x : ℝ) (y : ℝ) : f x = f y :=",
        ),
        (
            "theorem dummy (n : ℕ) (hn : n % 2 = 1) : 8 ∣ n^2 - 1 :=",
            "theorem dummy {n : ℕ} (hn : Odd n) : 8 ∣ (n^2 - 1) :=",
        ),
        (
            "theorem dumssmy {G : Type*} [Group G] (x : G) : x ^ 2 = 1 ↔ orderOf x = 1 ∨ orderOf x = 2 :=",
            "theorem dumssfmy {G : Type*} [Group G] : ∀ (x : G), orderOf x = 1 ∨ orderOf x = 2 ↔ x ^ 2 = 1 :=",
        ),
        (
            "theorem sP : Infinite {p : Nat.Primes // p ≡ -1 [ZMOD 6]} :=",
            "theorem sQ : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 6 = 5} :=",
        ),
        (
            "theorem dummy83 : Irreducible (Polynomial.C (12 : ℚ) + Polynomial.C (6 : ℚ) * Polynomial.X + Polynomial.X ^ 3) :=",
            "theorem dummy84 : Irreducible (12 + 6 * X + X ^ 3 : Polynomial ℚ) :=",
        ),
        (
            "theorem dummy90 {p : ℕ} (hp : Nat.Prime p) (n : ℕ) (hn : 0 < n) : Irreducible (Polynomial.C (1 : ℚ) * Polynomial.X ^ n - Polynomial.C (p : ℚ)) :=",
            "theorem dummy91 (p : ℕ) (hp : Prime p) (n : ℕ) (hn : n > 0) : Irreducible (X ^ n - (p : Polynomial ℚ) : Polynomial ℚ) :=",
        ),
        (
            "theorem dummy64 {X X' : Type*} [TopologicalSpace X] [TopologicalSpace X'] (π₁ : X × X' → X) (π₂ : X × X' → X') (h₁ : π₁ = Prod.fst) (h₂ : π₂ = Prod.snd) : IsOpenMap π₁ ∧ IsOpenMap π₂ :=",
            "theorem dummy63 {X X' : Type*} [TopologicalSpace X] [TopologicalSpace X'] : (∀ U : Set (X × X'), IsOpen U → IsOpen (Prod.fst '' U)) ∧ (∀ U : Set (X × X'), IsOpen U → IsOpen (Prod.snd '' U)) :=",
        ),
        (
            "theorem sP {R : Type u_1} [Ring R] (h : ∀ (x : R), x ^ 3 = x) (x : R) (y : R) : x * y = y * x :=",
            "theorem sQ {R : Type*} [Ring R] (h : ∀ x : R, x ^ 3 = x) : Nonempty (CommRing R) :=",
        ),
        (
            "theorem dummy {x : ℝ} (r : ℚ) (hr : r ≠ 0) (hx : Irrational x) : Irrational (r + x) :=",
            "theorem dummy (x : ℝ) (y : ℚ) (hy : y ≠ 0) : ( Irrational x ) -> Irrational ( x + y ) :=",
        ),
    ]

    console.print(f"{metric.__name__} metric on examples:")
    for formalization_1, formalization_2 in formalization_pairs:
        console.print()
        console.rule()
        console.print("Comparing formalizations:")
        console.print(Syntax(formalization_1, "lean4"))
        console.print(Syntax(formalization_2, "lean4"))
        console.print(
            f"Equivalent: {metric(formalization_1, formalization_2, src_header, repl_config, timeout_per_proof=20)}"
        )


def proofnetverif(metric):
    repl_config = LeanREPLConfig(lean_version="v4.8.0", require="mathlib")

    dataset = load_dataset("PAug/ProofNetVerif", split="train")

    nb_tested = 0
    nb_correct = 0
    for example in tqdm(dataset, desc=f"{metric.__name__} metric on ProofNetVerif dataset"):
        nb_tested += 1
        if metric(
            example["lean4_formalization"],
            example["prediction"],
            example["lean4_src_header"],
            repl_config,
            timeout_per_proof=20,
        ):
            nb_correct += 1

    console.print(f"{metric.__name__} metric on ProofNetVerif dataset: {nb_correct}/{nb_tested} correct")


if __name__ == "__main__":
    # metric = beql
    metric = beq_plus

    # all statements are equivalent, but some of these examples showcase failure cases of the BEq+ metric
    # examples_limitations(metric)

    proofnetverif(metric)
