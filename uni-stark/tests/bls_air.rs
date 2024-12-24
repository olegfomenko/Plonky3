use p3_air::{Air, AirBuilder, BaseAir};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger64};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, FieldAlgebra, PrimeField64};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::{DenseStorage, RowMajorMatrix};
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher32, SerializingHasher64,
    TruncatedPermutation,
};
use p3_uni_stark::SymbolicExpression::Variable;
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::thread_rng;
use std::marker::PhantomData;
use std::ops::Mul;

const q0: u64 = 725501752471715841;
const q1: u64 = 6461107452199829505;
const q2: u64 = 6968279316240510977;
const q3: u64 = 1345280370688173398;

const SZ: usize = 4;

/// For testing the public values feature
pub struct BLSAir {}

impl<F> BaseAir<F> for BLSAir {
    fn width(&self) -> usize {
        2 * SZ
    }
}

fn add64(x: u64, y: u64, carry: u64) -> (u64, u64) {
    let sum = x.wrapping_add(y).wrapping_add(carry);
    let carry_out = ((x & y) | ((x | y) & (!sum))) >> 63;
    (sum, carry_out)
}

fn smaller_than_modulus(z0: u64, z1: u64, z2: u64, z3: u64) -> bool {
    return (z3 < q3
        || (z3 == q3 && (z2 < q2 || (z2 == q2 && (z1 < q1 || (z1 == q1 && (z0 < q0)))))));
}

fn sub64(x: u64, y: u64, borrow: u64) -> (u64, u64) {
    let diff = x.wrapping_sub(y).wrapping_sub(borrow);
    let borrow_out = ((!x & y) | (!(x ^ y) & diff)) >> 63;
    (diff, borrow_out)
}

impl<AB: AirBuilder> Air<AB> for BLSAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let arg = main.row_slice(0); // get the current row
        let res = main.row_slice(1); // get the next row

        let mut z0: u64;
        let mut z1: u64;
        let mut z2: u64;
        let mut z3: u64;
        let mut calculated = false;

        let transition = || -> (u64, u64, u64, u64) {
            if !calculated {
                let x0 = arg[0].into().try_as_canonical_u64().unwrap();
                let x1 = arg[1].into().try_as_canonical_u64().unwrap();
                let x2 = arg[2].into().try_as_canonical_u64().unwrap();
                let x3 = arg[3].into().try_as_canonical_u64().unwrap();

                let y0 = arg[4].into().try_as_canonical_u64().unwrap();
                let y1 = arg[5].into().try_as_canonical_u64().unwrap();
                let y2 = arg[6].into().try_as_canonical_u64().unwrap();
                let y3 = arg[7].into().try_as_canonical_u64().unwrap();

                let mut carry: u64 = 0;

                (z0, carry) = add64(x0, y0, 0);
                (z1, carry) = add64(x1, y1, carry);
                (z2, carry) = add64(x2, y2, carry);
                (z3, _) = add64(x3, y3, carry);

                if !smaller_than_modulus(z0, z1, z2, z3) {
                    let mut b: u64 = 0;
                    (z0, b) = sub64(z0, q0, 0);
                    (z1, b) = sub64(z1, q1, b);
                    (z2, b) = sub64(z2, q2, b);
                    (z3, _) = sub64(z3, q3, b);
                }

                calculated = true;
            }

            return (z0, z1, z2, z3);
        };

        // Constrain the result
        builder
            .when_transition()
            .assert_eq(res[0], AB::Expr::from_canonical_u64(transition().0));
        builder
            .when_transition()
            .assert_eq(res[1], AB::Expr::from_canonical_u64(transition().1));
        builder
            .when_transition()
            .assert_eq(res[2], AB::Expr::from_canonical_u64(transition().2));
        builder
            .when_transition()
            .assert_eq(res[3], AB::Expr::from_canonical_u64(transition().3));
    }
}
//5398279529647893567
pub fn generate_trace_rows<F: PrimeField64>() -> RowMajorMatrix<F> {
    // x [8208527419514685428 15076030347479398436 5012346996662063847 1259127506789602721]
    // y [15636496183842759755 9516004183041941471 17583109545567259222 1212323145895467508]
    // z [4672777777176177726 18130927078321510403 15627177225988812092 1126170281996896831]
    const N: usize = 2;

    let mut values = F::zero_vec(N * SZ * 2);

    values[0] = F::from_canonical_u64(8208527419514685428);
    values[1] = F::from_canonical_u64(15076030347479398436);
    values[2] = F::from_canonical_u64(5012346996662063847);
    values[3] = F::from_canonical_u64(1259127506789602721);

    values[4] = F::from_canonical_u64(15636496183842759755);
    values[5] = F::from_canonical_u64(9516004183041941471);
    values[6] = F::from_canonical_u64(17583109545567259222);
    values[7] = F::from_canonical_u64(1212323145895467508);

    values[8] = F::from_canonical_u64(4672777777176177726);
    values[9] = F::from_canonical_u64(18130927078321510403);
    values[10] = F::from_canonical_u64(15627177225988812092);
    values[11] = F::from_canonical_u64(1126170281996896831);

    RowMajorMatrix::new(values, 2 * SZ)
}

// Your choice of Field
type Val = Goldilocks;

// This creates a cubic extension field over Val using a binomial basis. It's used for generating challenges in the proof system.
// The reason why we want to extend our field for Challenges, is because the original Field size is too small to be brute-forced to solve the challenge.
type Challenge = BinomialExtensionField<Val, 2>;

//
type Perm = Poseidon2Goldilocks<8>;

// Your choice of Hash Function
type MyHash = PaddingFreeSponge<Perm, 8, 4, 4>;

// Defines a compression function type using ByteHash, with 2 input blocks and 32-byte output.
type MyCompress = TruncatedPermutation<Perm, 2, 4, 8>;

// Defines a Merkle tree commitment scheme for field elements with 32 levels.
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 4>;

// Defines an extension of the Merkle tree commitment scheme for the challenge field.
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

type Dft = Radix2DitParallel<Val>;

// Defines the challenger type for generating random challenges.
type Challenger = DuplexChallenger<Val, Perm, 8, 4>;

// Defines the polynomial commitment scheme type.
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

// Defines the overall STARK configuration type.
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
#[test]
fn test_bls31_377_add() {
    let perm = Perm::new_from_rng_128(&mut thread_rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();

    // Configures the FRI (Fast Reed-Solomon IOP) protocol parameters.
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let config = MyConfig::new(pcs);

    // Generate the execution trace, based on the inputs defined above.
    let trace = generate_trace_rows();

    println!("trace: {:?}", trace);

    // Create Challenge sequence, in this case, we are using empty vector as seed inputs.
    let mut challenger = Challenger::new(perm.clone());

    let proof = prove(&config, &BLSAir {}, &mut challenger, trace, &vec![]);

    // Create the same Challenge sequence as above for verification purpose
    let mut challenger = Challenger::new(perm.clone());
    // Verify your proof!
    verify(&config, &BLSAir {}, &mut challenger, &proof, &vec![]).unwrap();
}
