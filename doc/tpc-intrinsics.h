//===--- tpc-intrinsics.h----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//-  
//-   Documentation presented in this file must be extracted by Doxygen or
//-   similar documentation tool to produce end users builtin functions
//-   reference.
//-
//-   Declarations presented in this file are used by TPC intrinsic generator to
//-   produce intrinsic definitions for TPC compiler. Syntax used here is a
//-   very restricted subset of C declaration syntax.
//------------------------------------------------------------------------------

typedef signed   int   int32_t;
typedef unsigned int   uint32_t;
typedef signed   short int16_t;
typedef unsigned short uint16_t;
typedef signed   char  int8_t;
typedef unsigned char  uint8_t;
typedef signed   char  int4_t;
typedef unsigned char  uint4_t;
typedef _BFloat16 bf16;

typedef struct _float64_int64_pair_t  float64_int64;
typedef struct _float64_uint64_pair_t  float64_uint64;
typedef struct _int64_float64_pair_t  int64_float64;
typedef struct _int64_uint64_pair_t  int64_uint64;
typedef struct _uint64_float64_pair_t  uint64_float64;
typedef struct _uint64_int64_pair_t  uint64_int64;
typedef struct _bfloat128_half128_pair_t  bfloat128_half128;
typedef struct _bfloat128_short128_pair_t  bfloat128_short128;
typedef struct _bfloat128_ushort128_pair_t  bfloat128_ushort128;
typedef struct _half128_bfloat128_pair_t  half128_bfloat128;
typedef struct _half128_short128_pair_t  half128_short128;
typedef struct _half128_ushort128_pair_t  half128_ushort128;
typedef struct _short128_bfloat128_pair_t  short128_bfloat128;
typedef struct _short128_half128_pair_t  short128_half128;
typedef struct _short128_ushort128_pair_t  short128_ushort128;
typedef struct _ushort128_bfloat128_pair_t  ushort128_bfloat128;
typedef struct _ushort128_half128_pair_t  ushort128_halft128;
typedef struct _ushort128_short128_pair_t  ushort128_short128;


/// @file
/// @brief Intrinsic functions
///
/// Most of the intrinsic functions represent particular instruction available
/// in TPC cores and are translated to single instruction.
///
/// Intrinsic names
/// ---------------
///
/// Intrinsic name has generally several components:
/// - scalar/vector prefix (`s_` or `v_`),
/// - operand type (like `f32_`),
/// - instruction mnemonic (like `add`),
/// - switch indicator (like `_acc32`),
/// - distinguishing suffix (like `_vb`).
/// All of them except instruction mnemonic are optional.
///
/// Scalar/vector prefix helps to distinguish between instruction that do the
/// same operation but for scalar and vector data, like `s_i32_add` and
/// `v_i32_add_b`.
///
/// Operand type designates type of operands of the corresponding instruction.
/// In general it should not be considered as the return value type, although in
/// many cases it is so. In some cases operand type is encoded in the instruction
/// mnemonic, for convenience or for backward compatibility, as, for instance, in
/// `s_convert_i8_to_bf16`. So it may absent even in instructions for which
/// operand type is applicable.
///
/// Instruction mnemonic is same as the corresponding instruction name. In some
/// cases the mnemonic is amended with additional information that helps to
/// represent instruction. One example is the aforementioned 'convert' intrinsic.
/// Another case is SEL instructions. Corresponding intrinsics have names like
/// `v_f32_sel_eq_i32_b`. Type are provided to mimic function result and arguments.
/// This instruction compares values of type 'vector of i32' ('arguments') and
/// returns value of type `f32` ('result').
///
/// Switches usually are encoded in a separate argument. However there are
/// switches that change function signature. For instance, if MAC instruction
/// has switch `ACC_F32`, corresponding intrinsic must return `float128`, not
/// `bfloat128` as in the absence of this switch. Different functions must
/// represent the instruction with and without the switch. In such cases the
/// switch is encoded in the intrinsic name: `s_bf16_mac` and `s_bf16_mac_acc32`.
///
/// Distinguishing suffix allows to have different names for the same instruction
/// but with different arguments, line `v_add_b` and `v_add_vb`. In many cases these
/// suffixes are `_b` and `_vb` if instruction allows both scalar and vector
/// predicates.
///
/// Intrinsic arguments
/// -------------------
///
/// The set of argument accepted by an intrinsic is determined by the operands of
/// corresponding instruction. Most intrinsics have the arguments:
/// - switch set,
/// - income value,
/// - predicate,
/// - polarity.
/// They are used just in this sequence at the end of argument list, like:
///
/// @code
///     int32_t s_i32_add(int32_t a, int32_t b, int switches, int32_t income, bool predicate, bool polarity);
/// @endcode
///
/// Switch set represent modifiers provided for instruction. It must be an
/// integer value, known at compile time. It is a combination of constants, like:
///
/// @code
///     s_i32_sub(a, b, SW_SAT | SW_NEG, 0, 1, 0);
/// @endcode
///
/// Each intrinsic allows its own set of switches. If no special switches are
/// needed, zero value should be used.
/// Some switches change type or number of intrinsic arguments, for example,
/// ACC_F32, PARTIAL, TO_16. They cannot be specified in switch set
/// argument, to use them another intrinsic should be used, for instance,
/// `s_bf16_mac_acc32` instead of `s_bf16_mac`.
///
/// The income attribute represent the income value of DEST ('destination')
/// to be updated by the instruction.
/// Predicate, polarity and income value are needed to support conditional
/// execution. Almost every instruction can be executed conditionally
/// depending on the value of some predicate. Intrinsic as a C function must
/// always return appropriate value even if the instruction is not executed. The
/// value returned in this case is determined by income argument.
/// Hypothetical intrinsic for add operation may be represented in pseudocode as:
/// @code
///     int add(int a, int b, int income, bool predicate, bool polarity) {
///       if (predicate != polarity)
///         return a + b;
///       else
///         return income;
///     }
/// @endcode
/// For intrinsic function that receive vector predicate the returned vector will
/// be composed from instruction result elements where predicate == true,
/// and income value elements where predicate == false.
///
/// Some instructions (like SET_INDX or MAC) do not produce new value, but modify
/// some of its input arguments. In these cases separate input value is not
/// required, as the intrinsic may return the argument to modify. Such intrinsics do
/// not have input argument.
///

//
// ------ High-level intrinsics
//

/// @brief Converts value of some type to another type.
///
/// @param a        The value to convert.
/// @param options  OR'ed set of constants. \n
///                 Only constants that represent rounding mode are supported now. \n
///                 They are the same constants, that are used as switches in CONVERT intrinsics.
/// @return Converted value
///
/// @{
float64 convert_int64_to_float64(int64 a, const int options);
float64 convert_uint64_to_float64(uint64 a, const int options);

float128 convert_bfloat128_to_float128(bfloat128 a, const int options);
float128 convert_half128_to_float128(half128 a, const int options);
float128 convert_int128_to_float128(int128 a, const int options);
float128 convert_uint128_to_float128(uint128 a, const int options);
float128 convert_short128_to_float128(short128 a, const int options);
float128 convert_ushort128_to_float128(ushort128 a, const int options);

float256 convert_bfloat256_to_float256(bfloat256 a, const int options);
float256 convert_half256_to_float256(half256 a, const int options);
float256 convert_int256_to_float256(int256 a, const int options);
float256 convert_uint256_to_float256(uint256 a, const int options);
float256 convert_short256_to_float256(short256 a, const int options);
float256 convert_ushort256_to_float256(ushort256 a, const int options);
float256 convert_char256_to_float256(char256 a, const int options);
float256 convert_uchar256_to_float256(uchar256 a, const int options);

bfloat128 convert_float128_to_bfloat128(float128 a, const int options);
bfloat128 convert_half128_to_bfloat128(half128 a, const int options);
bfloat128 convert_int128_to_bfloat128(int128 a, const int options);
bfloat128 convert_uint128_to_bfloat128(uint128 a, const int options);
bfloat128 convert_short128_to_bfloat128(short128 a, const int options);
bfloat128 convert_ushort128_to_bfloat128(ushort128 a, const int options);

bfloat256 convert_float256_to_bfloat256(float256 a, const int options);
bfloat256 convert_half256_to_bfloat256(half256 a, const int options);
bfloat256 convert_int256_to_bfloat256(int256 a, const int options);
bfloat256 convert_uint256_to_bfloat256(uint256 a, const int options);
bfloat256 convert_short256_to_bfloat256(short256 a, const int options);
bfloat256 convert_ushort256_to_bfloat256(ushort256 a, const int options);
bfloat256 convert_char256_to_bfloat256(char256 a, const int options);
bfloat256 convert_uchar256_to_bfloat256(uchar256 a, const int options);

half128 convert_float128_to_half128(float128 a, const int options);
half128 convert_bfloat128_to_half128(bfloat128 a, const int options);
half128 convert_int128_to_half128(int128 a, const int options);
half128 convert_uint128_to_half128(uint128 a, const int options);
half128 convert_short128_to_half128(short128 a, const int options);
half128 convert_ushort128_to_half128(ushort128 a, const int options);

half256 convert_float256_to_half256(float256 a, const int options);
half256 convert_bfloat256_to_half256(bfloat256 a, const int options);
half256 convert_int256_to_half256(int256 a, const int options);
half256 convert_uint256_to_half256(uint256 a, const int options);
half256 convert_short256_to_half256(short256 a, const int options);
half256 convert_ushort256_to_half256(ushort256 a, const int options);





int64 convert_float64_to_int64(float64 a, const int options);
int64 convert_uint64_to_int64(uint64 a, const int options);

int128 convert_float128_to_int128(float128 a, const int options);
int128 convert_bfloat128_to_int128(bfloat128 a, const int options);
int128 convert_half128_to_int128(half128 a, const int options);
int128 convert_uint128_to_int128(uint128 a, const int options);
int128 convert_short128_to_int128(short128 a, const int options);
int128 convert_ushort128_to_int128(ushort128 a, const int options);

int256 convert_float256_to_int256(float256 a, const int options);
int256 convert_bfloat256_to_int256(bfloat256 a, const int options);
int256 convert_half256_to_int256(half256 a, const int options);
int256 convert_uint256_to_int256(uint256 a, const int options);
int256 convert_short256_to_int256(short256 a, const int options);
int256 convert_ushort256_to_int256(ushort256 a, const int options);
int256 convert_char256_to_int256(char256 a, const int options);
int256 convert_uchar256_to_int256(uchar256 a, const int options);

uint64 convert_float64_to_uint64(float64 a, const int options);
uint64 convert_int64_to_uint64(int64 a, const int options);

uint128 convert_float128_to_uint128(float128 a, const int options);
uint128 convert_bfloat128_to_uint128(bfloat128 a, const int options);
uint128 convert_half128_to_uint128(half128 a, const int options);
uint128 convert_int128_to_uint128(int128 a, const int options);
uint128 convert_short128_to_uint128(short128 a, const int options);
uint128 convert_ushort128_to_uint128(ushort128 a, const int options);

uint256 convert_float256_to_uint256(float256 a, const int options);
uint256 convert_bfloat256_to_uint256(bfloat256 a, const int options);
uint256 convert_half256_to_uint256(half256 a, const int options);
uint256 convert_int256_to_uint256(int256 a, const int options);
uint256 convert_short256_to_uint256(short256 a, const int options);
uint256 convert_ushort256_to_uint256(ushort256 a, const int options);
uint256 convert_char256_to_uint256(char256 a, const int options);
uint256 convert_uchar256_to_uint256(uchar256 a, const int options);

short128 convert_float128_to_short128(float128 a, const int options);
short128 convert_bfloat128_to_short128(bfloat128 a, const int options);
short128 convert_half128_to_short128(half128 a, const int options);
short128 convert_int128_to_short128(int128 a, const int options);
short128 convert_uint128_to_short128(uint128 a, const int options);
short128 convert_ushort128_to_short128(ushort128 a, const int options);

short256 convert_float256_to_short256(float256 a, const int options);
short256 convert_bfloat256_to_short256(bfloat256 a, const int options);
short256 convert_half256_to_short256(half256 a, const int options);
short256 convert_int256_to_short256(int256 a, const int options);
short256 convert_uint256_to_short256(uint256 a, const int options);
short256 convert_ushort256_to_short256(ushort256 a, const int options);
short256 convert_char256_to_short256(char256 a, const int options);
short256 convert_uchar256_to_short256(uchar256 a, const int options);

ushort128 convert_float128_to_ushort128(float128 a, const int options);
ushort128 convert_bfloat128_to_ushort128(bfloat128 a, const int options);
ushort128 convert_half128_to_ushort128(half128 a, const int options);
ushort128 convert_int128_to_ushort128(int128 a, const int options);
ushort128 convert_uint128_to_ushort128(uint128 a, const int options);
ushort128 convert_short128_to_ushort128(short128 a, const int options);

ushort256 convert_float256_to_ushort256(float256 a, const int options);
ushort256 convert_bfloat256_to_ushort256(bfloat256 a, const int options);
ushort256 convert_half256_to_ushort256(half256 a, const int options);
ushort256 convert_int256_to_ushort256(int256 a, const int options);
ushort256 convert_uint256_to_ushort256(uint256 a, const int options);
ushort256 convert_short256_to_ushort256(short256 a, const int options);
ushort256 convert_char256_to_ushort256(char256 a, const int options);
ushort256 convert_uchar256_to_ushort256(uchar256 a, const int options);

char256 convert_float256_to_char256(float256 a, const int options);
char256 convert_bfloat256_to_char256(bfloat256 a, const int options);
char256 convert_half256_to_char256(half256 a, const int options);
char256 convert_int256_to_char256(int256 a, const int options);
char256 convert_uint256_to_char256(uint256 a, const int options);
char256 convert_short256_to_char256(short256 a, const int options);
char256 convert_ushort256_to_char256(ushort256 a, const int options);
char256 convert_uchar256_to_char256(uchar256 a, const int options);

uchar256 convert_float256_to_uchar256(float256 a, const int options);
uchar256 convert_bfloat256_to_uchar256(bfloat256 a, const int options);
uchar256 convert_half256_to_uchar256(half256 a, const int options);
uchar256 convert_int256_to_uchar256(int256 a, const int options);
uchar256 convert_uint256_to_uchar256(uint256 a, const int options);
uchar256 convert_short256_to_uchar256(short256 a, const int options);
uchar256 convert_ushort256_to_uchar256(ushort256 a, const int options);
uchar256 convert_char256_to_uchar256(char256 a, const int options);
/// @}


//
// ------ ABS
//

/// @brief Represents ABS instruction - Calculates absolute value.
///
/// @param a           The argument (SRC1).
/// @param switches    Switches of ABS instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Absolute value of operand \p a.
///
/// This operation is implemented as instruction ABS for integer operands and \n
/// FORM_FP_NUMBER for floating point (actually AND on SPU).
///
///
/// @{
float s_f32_abs(float a, int switches=0, float income=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_abs(bf16 a, int switches=0, bf16 income=0, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_abs(int32_t a, int switches=0, int32_t income=0, bool predicate=1, bool polarity=0);
int16_t s_i16_abs(int16_t a, int switches=0, int16_t income=0, bool predicate=1, bool polarity=0);
int8_t s_i8_abs(int8_t a, int switches=0, int8_t income=0, bool predicate=1, bool polarity=0);
float64 v_f32_abs_b(float64 a, int switches=0, float64 income=0, bool predicate=1, bool polarity=0);
float64 v_f32_abs_vb(float64 a, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_abs_b(bfloat128 a, int switches=0, bfloat128 income=0, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_abs_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
int64 v_i32_abs_b(int64 a, int switches=0, int64 income=0, bool predicate=1, bool polarity=0);
int64 v_i32_abs_vb(int64 a, int switches, int64 income, bool64 predicate, bool polarity=0);
short128 v_i16_abs_b(short128 a, int switches=0, short128 income=0, bool predicate=1, bool polarity=0);
short128 v_i16_abs_vb(short128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
char256 v_i8_abs_b(char256 a, int switches=0, char256 income=0, bool predicate=1, bool polarity=0);
char256 v_i8_abs_vb(char256 a, int switches, char256 income, bool256 predicate, bool polarity=0);
/// @}

/// @brief Represents ABS instruction for int5 operands.
///
/// @param a           The argument (SRC1).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of ABS instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return IRF of absolute values of argument \p a.
///
///
/// @{
int5 i_i32_abs(int5 a, const int dimmask, const int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ ADD
//

/// @brief Represents ADD instruction.
///
/// @param a           The first SRC operand to ADD (SRC1).
/// @param b           The second SRC operand to ADD (SRC2).
/// @param switches    Switches of ADD instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Sum of operands \p a and \p b.
///
/// \par Allowed switches are:
///     \li SW_SAT - Saturate (integer types only). \n
///
/// @{
float s_f32_add(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_add(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_add(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_add(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_add(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_add(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_add(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_add(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
float64 v_f32_add_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_add_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_add_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_add_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_add_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_add_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_add_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_add_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_add_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_add_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_add_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_add_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_add_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_add_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_add_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_add_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
/// @}

/// @brief Represents scalar ADD instruction for integer operands in index registers.
///
/// @param a           The first SRC operand to ADD (SRC1).
/// @param b           The second SRC operand to ADD (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of ADD instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of ADD operation between \p a and \p b.
///
/// @{
int5 i_i32_add(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ AND
//

/// @brief Represents AND instruction.
///
/// @param a           The first SRC operand to AND (SRC1).
/// @param b           The second SRC operand to AND (SRC2).
/// @param switches    Switches of AND instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of AND operation between \p a and \p b.
///
///
/// @{
float s_f32_and(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_and(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_and(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_and(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_and(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_and(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_and(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_and(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_and(bool a, bool b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
float64 v_f32_and_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_and_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_and_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_and_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_and_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_and_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_and_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_and_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_and_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_and_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_and_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_and_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_and_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_and_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_and_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_and_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_and_b(bool256 a, bool256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_and_vb(bool256 a, bool256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
/// @}

/// @brief Represents AND instruction for int5.
///
/// @param a           The first SRC operand to AND (SRC1).
/// @param b           The second SRC operand to AND (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of AND instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of AND operation between \p a and \p b.
///
/// @{
int5 i_i32_and(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ ASH
//

/// @brief Represents ASH instruction.
///
/// @param a           The value to shift (SRC1).
/// @param b           The number of bits to shift (SRC2).
/// @param switches    Switches of ASH instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation - \c DEST=a*(2^b).
///
/// \par Allowed switches are:
///     \li SW_SAT - Saturate (should always be set, left shift saturation is enabled by default).
///     \li SW_RHU - Round half up.
///
/// @{
int32_t s_i32_ash(int32_t a, int8_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_ash(uint32_t a, int8_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_ash(int16_t a, int8_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_ash(uint16_t a, int8_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_ash(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_ash(uint8_t a, int8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ash_b(int64 a, char256 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ash_vb(int64 a, char256 b, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ash_b(uint64 a, char256 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ash_vb(uint64 a, char256 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_ash_b(short128 a, char256 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ash_vb(short128 a, char256 b, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ash_b(ushort128 a, char256 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ash_vb(ushort128 a, char256 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_ash_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ash_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ash_b(uchar256 a, char256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ash_vb(uchar256 a, char256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}

//
// ------ ASO
//

/// @brief Represents ASO instruction - Atomic semaphore operation.
///
/// @param switches    Switches of ASO instructions.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
/// \par Allowed switches are:
///     \li SW_INC - Increment the semaphore value once all older writes to memory are observable.
#if defined(__gaudi_plus__)
///     \li SW_DEC - Decrement the semaphore value once all older writes to memory are observable.
#endif
///     \li SW_SPU - Semaphore update is done in SPU stage.
///     \li SW_VPU - Semaphore update is done in VPU stage.
///
/// @{
void aso(int switches=0, bool predicate=1, bool polarity=0);
/// @}


//
// ------ BREV
//

#if defined(__gaudi_plus__)
/// @brief Represents BREV instruction.
///
/// @param a           The source operand (SRC1).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The source operand bits in reversed order.
///
#endif
/// @{
#if defined(__gaudi_plus__)
uint32_t s_f32_brev(float a, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_bf16_brev(bf16 a, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int32_t s_i32_brev(int32_t a, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_brev(uint32_t a, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_brev(int16_t a, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_brev(uint16_t a, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_brev(int8_t a, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_brev(uint8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
uint64 v_f32_brev_b(float64 a, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_f32_brev_vb(float64 a, int switches, uint64 income, bool64 predicate, bool polarity=0);
ushort128 v_bf16_brev_b(bfloat128 a, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_bf16_brev_vb(bfloat128 a, int switches, ushort128 income, bool128 predicate, bool polarity=0);
int64 v_i32_brev_b(int64 a, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_brev_vb(int64 a, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_brev_b(uint64 a, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_brev_vb(uint64 a, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_brev_b(short128 a, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_brev_vb(short128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_brev_b(ushort128 a, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_brev_vb(ushort128 a, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_brev_b(char256 a, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_brev_vb(char256 a, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_brev_b(uchar256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_brev_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
#endif
/// @}


//
// ------ CACHE_FLUSH
//

/// @brief Represents CACHE_FLUSH instruction - Flush the Data-cache.
///
/// @param switches    Switches of the instructions.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
/// @{
void cache_flush(int switches=0, bool predicate=1, bool polarity=0);
/// @}


//
// ------ CACHE_INVALIDATE
//

/// @brief Represents CACHE_INVALIDATE instruction - Invalidate Data/LUT/SB cache
///
/// @param switches    Switches of the instructions.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
///
/// @{
void cache_invalidate(int switches=0, bool predicate=1, bool polarity=0);
/// @}


//
// ------ CALC_FP_SPECIAL
//

#if defined(__gaudi_plus__)
/// @brief Represents CALC_FP_SPECIAL instruction.
///
/// @param src1        The first operand (SRC1).
/// @param src2        The second SRC operand (SRC2). \n
///                    For unary functions it is ignored and may be of any value, for instance, zero. \n
///                    \p src1 and \p src2 are 8/10bit-masks format (vector or scalar) representing the \n
///                    class of the operands - returned value of FCLASS instruction.
/// @param switches    Code of the function.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Calculate the choosen function for all special class elements. \n
///         (for non special class elements returns income).
///
/// \par Allowed switches are:
///     - [SPECIAL_FUNC]
///         \li SW_RECIP - reciprocal - \c 1/x
///         \li SW_RSQRT - reciprocal square root of x
///         \li SW_SQRT - square root of x
///         \li SW_LOG - logarithm of x
///         \li SW_EXP - exponent - \c e^x
///         \li SW_TANH - hyperbolic tangent of x
///         \li SW_DIV - divide - \c x/y
///         \li SW_POW - power - \c x^y
///
#endif
/// @{
#if defined(__gaudi_plus__)
float s_f32_calc_fp_special(float src1, float src2, int switches, float income, bool predicate=1, bool polarity=0);
bf16 s_bf16_calc_fp_special(bf16 src1, bf16 src2, int switches, bf16 income, bool predicate=1, bool polarity=0);
float64 v_f32_calc_fp_special_b(float64 src1, float64 src2, int switches, float64 income, bool predicate=1, bool polarity=0);
float64 v_f32_calc_fp_special_vb(float64 src1, float64 src2, int switches, float64 income, bool64 predicate, bool polarity=0);
bfloat128 v_bf16_calc_fp_special_b(bfloat128 src1, bfloat128 src2, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_calc_fp_special_vb(bfloat128 src1, bfloat128 src2, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
/// @}


//
// ------ CMP_EQ
//

/// @brief Represents CMP_EQ instruction.
///
/// @param a           The first SRC operand (SRC1).
/// @param b           The second SRC operand (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Boolean result of comparison - \c a==b (scalar or vector).
///
#if defined(__gaudi_plus__)
/// \par Allowed switches are:
///     \li SW_MASK_EQ_ZERO - Compare between (a & b) and 0.
#endif
///
/// @{
bool s_f32_cmp_eq(float a, float b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool s_bf16_cmp_eq(bf16 a, bf16 b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#endif
bool s_i32_cmp_eq(int32_t a, int32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u32_cmp_eq(uint32_t a, uint32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i16_cmp_eq(int16_t a, int16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u16_cmp_eq(uint16_t a, uint16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i8_cmp_eq(int8_t a, int8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u8_cmp_eq(uint8_t a, uint8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

bool64 v_f32_cmp_eq_vb(float64 a, float64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_f32_cmp_eq_b(float64 a, float64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool128 v_bf16_cmp_eq_vb(bfloat128 a, bfloat128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_bf16_cmp_eq_b(bfloat128 a, bfloat128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
#endif
bool64 v_i32_cmp_eq_vb(int64 a, int64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_i32_cmp_eq_b(int64 a, int64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool64 v_u32_cmp_eq_vb(uint64 a, uint64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_u32_cmp_eq_b(uint64 a, uint64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool128 v_i16_cmp_eq_vb(short128 a, short128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_i16_cmp_eq_b(short128 a, short128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool128 v_u16_cmp_eq_vb(ushort128 a, ushort128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_u16_cmp_eq_b(ushort128 a, ushort128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool256 v_i8_cmp_eq_vb(char256 a, char256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i8_cmp_eq_b(char256 a, char256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_u8_cmp_eq_vb(uchar256 a, uchar256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_u8_cmp_eq_b(uchar256 a, uchar256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}

//
// ------ CMP_GEQ
//


/// @brief Represents CMP_GEQ instruction.
///
/// @param a           The first SRC operand (SRC1).
/// @param b           The second SRC operand (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Boolean result of comparison - \c a>=b (scalar or vector).
///
/// @{
bool s_f32_cmp_geq(float a, float b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool s_bf16_cmp_geq(bf16 a, bf16 b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#endif
bool s_i32_cmp_geq(int32_t a, int32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u32_cmp_geq(uint32_t a, uint32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i16_cmp_geq(int16_t a, int16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u16_cmp_geq(uint16_t a, uint16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i8_cmp_geq(int8_t a, int8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u8_cmp_geq(uint8_t a, uint8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

bool64 v_f32_cmp_geq_vb(float64 a, float64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_f32_cmp_geq_b(float64 a, float64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool128 v_bf16_cmp_geq_vb(bfloat128 a, bfloat128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_bf16_cmp_geq_b(bfloat128 a, bfloat128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
#endif
bool64 v_i32_cmp_geq_vb(int64 a, int64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_i32_cmp_geq_b(int64 a, int64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool64 v_u32_cmp_geq_vb(uint64 a, uint64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_u32_cmp_geq_b(uint64 a, uint64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool128 v_i16_cmp_geq_vb(short128 a, short128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_i16_cmp_geq_b(short128 a, short128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool128 v_u16_cmp_geq_vb(ushort128 a, ushort128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_u16_cmp_geq_b(ushort128 a, ushort128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool256 v_i8_cmp_geq_vb(char256 a, char256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i8_cmp_geq_b(char256 a, char256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_u8_cmp_geq_vb(uchar256 a, uchar256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_u8_cmp_geq_b(uchar256 a, uchar256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}

//
// ------ CMP_GRT
//


/// @brief Represents CMP_GRT instruction.
///
/// @param a           The first SRC operand (SRC1).
/// @param b           The second SRC operand (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Boolean result of comparison - \c a>b (scalar or vector).
///
/// @{
bool s_f32_cmp_grt(float a, float b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool s_bf16_cmp_grt(bf16 a, bf16 b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#endif
bool s_i32_cmp_grt(int32_t a, int32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u32_cmp_grt(uint32_t a, uint32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i16_cmp_grt(int16_t a, int16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u16_cmp_grt(uint16_t a, uint16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i8_cmp_grt(int8_t a, int8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u8_cmp_grt(uint8_t a, uint8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

bool64 v_f32_cmp_grt_vb(float64 a, float64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_f32_cmp_grt_b(float64 a, float64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool128 v_bf16_cmp_grt_vb(bfloat128 a, bfloat128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_bf16_cmp_grt_b(bfloat128 a, bfloat128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
#endif
bool64 v_i32_cmp_grt_vb(int64 a, int64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_i32_cmp_grt_b(int64 a, int64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool64 v_u32_cmp_grt_vb(uint64 a, uint64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_u32_cmp_grt_b(uint64 a, uint64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool128 v_i16_cmp_grt_vb(short128 a, short128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_i16_cmp_grt_b(short128 a, short128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool128 v_u16_cmp_grt_vb(ushort128 a, ushort128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_u16_cmp_grt_b(ushort128 a, ushort128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool256 v_i8_cmp_grt_vb(char256 a, char256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i8_cmp_grt_b(char256 a, char256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_u8_cmp_grt_vb(uchar256 a, uchar256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_u8_cmp_grt_b(uchar256 a, uchar256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}

//
// ------ CMP_LEQ
//


/// @brief Represents CMP_LEQ instruction.
///
/// @param a           The first SRC operand (SRC1).
/// @param b           The second SRC operand (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Boolean result of comparison - \c a<=b (scalar or vector).
///
/// @{
bool s_f32_cmp_leq(float a, float b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool s_bf16_cmp_leq(bf16 a, bf16 b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#endif

bool s_i32_cmp_leq(int32_t a, int32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u32_cmp_leq(uint32_t a, uint32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i16_cmp_leq(int16_t a, int16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u16_cmp_leq(uint16_t a, uint16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i8_cmp_leq(int8_t a, int8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u8_cmp_leq(uint8_t a, uint8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

bool64 v_f32_cmp_leq_vb(float64 a, float64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_f32_cmp_leq_b(float64 a, float64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool128 v_bf16_cmp_leq_vb(bfloat128 a, bfloat128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_bf16_cmp_leq_b(bfloat128 a, bfloat128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
#endif
bool64 v_i32_cmp_leq_vb(int64 a, int64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_i32_cmp_leq_b(int64 a, int64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool64 v_u32_cmp_leq_vb(uint64 a, uint64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_u32_cmp_leq_b(uint64 a, uint64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool128 v_i16_cmp_leq_vb(short128 a, short128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_i16_cmp_leq_b(short128 a, short128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool128 v_u16_cmp_leq_vb(ushort128 a, ushort128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_u16_cmp_leq_b(ushort128 a, ushort128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool256 v_i8_cmp_leq_vb(char256 a, char256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i8_cmp_leq_b(char256 a, char256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_u8_cmp_leq_vb(uchar256 a, uchar256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_u8_cmp_leq_b(uchar256 a, uchar256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}

//
// ------ CMP_LESS
//


/// @brief Represents CMP_LESS instruction.
///
/// @param a           The first SRC operand (SRC1).
/// @param b           The second SRC operand (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Boolean result of comparison - \c a<b (scalar or vector).
///
/// @{
bool s_f32_cmp_less(float a, float b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool s_bf16_cmp_less(bf16 a, bf16 b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#endif
bool s_i32_cmp_less(int32_t a, int32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u32_cmp_less(uint32_t a, uint32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i16_cmp_less(int16_t a, int16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u16_cmp_less(uint16_t a, uint16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i8_cmp_less(int8_t a, int8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u8_cmp_less(uint8_t a, uint8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

bool64 v_f32_cmp_less_vb(float64 a, float64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_f32_cmp_less_b(float64 a, float64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool128 v_bf16_cmp_less_vb(bfloat128 a, bfloat128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_bf16_cmp_less_b(bfloat128 a, bfloat128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
#endif
bool64 v_i32_cmp_less_vb(int64 a, int64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_i32_cmp_less_b(int64 a, int64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool64 v_u32_cmp_less_vb(uint64 a, uint64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_u32_cmp_less_b(uint64 a, uint64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool128 v_i16_cmp_less_vb(short128 a, short128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_i16_cmp_less_b(short128 a, short128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool128 v_u16_cmp_less_vb(ushort128 a, ushort128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_u16_cmp_less_b(ushort128 a, ushort128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool256 v_i8_cmp_less_vb(char256 a, char256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i8_cmp_less_b(char256 a, char256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_u8_cmp_less_vb(uchar256 a, uchar256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_u8_cmp_less_b(uchar256 a, uchar256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}

//
// ------ CMP_NEQ
//


/// @brief Represents CMP_NEQ instruction.
///
/// @param a           The first SRC operand (SRC1).
/// @param b           The second SRC operand (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Boolean result of comparison - \c a!=b (scalar or vector).
///
/// @{
bool s_f32_cmp_neq(float a, float b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool s_bf16_cmp_neq(bf16 a, bf16 b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#endif
bool s_i32_cmp_neq(int32_t a, int32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u32_cmp_neq(uint32_t a, uint32_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i16_cmp_neq(int16_t a, int16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u16_cmp_neq(uint16_t a, uint16_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_i8_cmp_neq(int8_t a, int8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
bool s_u8_cmp_neq(uint8_t a, uint8_t b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

bool64 v_f32_cmp_neq_vb(float64 a, float64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_f32_cmp_neq_b(float64 a, float64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bool128 v_bf16_cmp_neq_vb(bfloat128 a, bfloat128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_bf16_cmp_neq_b(bfloat128 a, bfloat128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
#endif
bool64 v_i32_cmp_neq_vb(int64 a, int64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_i32_cmp_neq_b(int64 a, int64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool64 v_u32_cmp_neq_vb(uint64 a, uint64 b, int switches, bool64 income, bool64 predicate, bool polarity=0);
bool64 v_u32_cmp_neq_b(uint64 a, uint64 b, int switches=0, bool64 income={}, bool predicate=1, bool polarity=0);
bool128 v_i16_cmp_neq_vb(short128 a, short128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_i16_cmp_neq_b(short128 a, short128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool128 v_u16_cmp_neq_vb(ushort128 a, ushort128 b, int switches, bool128 income, bool128 predicate, bool polarity=0);
bool128 v_u16_cmp_neq_b(ushort128 a, ushort128 b, int switches=0, bool128 income={}, bool predicate=1, bool polarity=0);
bool256 v_i8_cmp_neq_vb(char256 a, char256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i8_cmp_neq_b(char256 a, char256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_u8_cmp_neq_vb(uchar256 a, uchar256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_u8_cmp_neq_b(uchar256 a, uchar256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}

//
// ------ CONVERT
//


/// @brief Represents CONVERT instruction in scalar slot.
///
/// @param src       The value to convert (SRC1).
/// @param switches  Switches of the instructions.
/// @param income    Income value of DEST.
/// @param predicate Predicate value for the instruction.
/// @param polarity  True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
#if defined(__gaudi_plus__)
///         \li SW_SR - Stochastic Rounding.
///         \li SW_RHAZ - Round half away from zero.
#endif
///         \li SW_CSR - Take rounding mode from
#if defined(__gaudi__) || defined(__goya__)
///                    ROUND_CSR register
#endif
///
/// @{
int32_t s_convert_f32_to_i32(float src, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_convert_f32_to_i16(float src, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_convert_f32_to_i8(float src, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_convert_f32_to_bf16(float src, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
float s_convert_bf16_to_f32(bf16 src, int switches=0, float income={}, bool predicate=1, bool polarity=0);
int16_t s_convert_bf16_to_i16(bf16 src, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
#endif
float s_convert_i32_to_f32(int32_t src, int switches=0, float income={}, bool predicate=1, bool polarity=0);
uint32_t s_convert_i32_to_u32(int32_t src, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
#if defined(__goya__)
uint8_t s_convert_i32_to_u8(int32_t src, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
bf16 s_convert_i32_to_bf16(int32_t src, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
float s_convert_i16_to_f32(int16_t src, int switches=0, float income={}, bool predicate=1, bool polarity=0);
int32_t s_convert_i16_to_i32(int16_t src, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_convert_i16_to_u32(int16_t src, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_convert_i16_to_u16(int16_t src, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_convert_i16_to_u8(int16_t src, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_convert_i16_to_bf16(int16_t src, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
float s_convert_i8_to_f32(int8_t src, int switches=0, float income={}, bool predicate=1, bool polarity=0);
int32_t s_convert_i8_to_i32(int8_t src, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_convert_i8_to_u32(int8_t src, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_convert_i8_to_i16(int8_t src, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_convert_i8_to_u16(int8_t src, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_convert_i8_to_u8(int8_t src, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_convert_i8_to_bf16(uint8_t src, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
bf16 s_convert_u16_to_bf16(uint16_t src, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


/// @brief Represents CONVERT instruction in vector slot, in which the result \n
/// vector element is of the same size as the source one.
///
/// @param src       The value to convert (SRC1).
/// @param switches  Switches of the instructions.
/// @param income    Income value of DEST.
/// @param predicate Predicate value for the instruction.
/// @param polarity  True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
#if defined(__gaudi_plus__)
///         \li SW_SR - Stochastic Rounding.
///         \li SW_RHAZ - Round half away from zero.
#endif
///         \li SW_CSR - Take rounding mode from
#if defined(__gaudi__) || defined(__goya__)
///                    ROUND_CSR register
#endif
///
/// @{
int64 v_convert_f32_to_i32_b(float64 src, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_convert_f32_to_i32_vb(float64 src, int switches, int64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
short128 v_convert_bf16_to_i16_b(bfloat128 src, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_convert_bf16_to_i16_vb(bfloat128 src, int switches, short128 income, bool128 predicate, bool polarity=0);
#endif
float64 v_convert_i32_to_f32_b(int64 src, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_convert_i32_to_f32_vb(int64 src, int switches, float64 income, bool64 predicate, bool polarity=0);
uint64 v_convert_i32_to_u32_b(int64 src, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_convert_i32_to_u32_vb(int64 src, int switches, uint64 income, bool64 predicate, bool polarity=0);
ushort128 v_convert_i16_to_u16_b(short128 src, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_convert_i16_to_u16_vb(short128 src, int switches, ushort128 income, bool128 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_convert_i16_to_bf16_b(short128 src, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_convert_i16_to_bf16_vb(short128 src, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
bfloat128 v_convert_u16_to_bf16_b(ushort128 src, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_convert_u16_to_bf16_vb(ushort128 src, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
uchar256 v_convert_i8_to_u8_b(char256 src, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_convert_i8_to_u8_vb(char256 src, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


/// @brief Represents CONVERT instruction in vector slot in which result vector \n
/// element is wider then the source one (up-convert).
///
/// @param src       The value to convert (SRC1).
/// @param switches  Switches of the instructions.
/// @param income    Income value of DEST.
/// @param predicate Predicate value for the instruction.
/// @param polarity  True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
#if defined(__gaudi_plus__)
///         \li SW_SR - Stochastic Rounding.
///         \li SW_RHAZ - Round half away from zero.
#endif
///         \li SW_CSR - Take rounding mode from
#if defined(__gaudi__) || defined(__goya__)
///                    ROUND_CSR register
#endif
///
/// @{
float64 v_convert_i16_to_f32_b(short128 src, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_convert_i16_to_f32_vb(short128 src, int switches, float64 income, bool64 predicate, bool polarity=0);
int64 v_convert_i16_to_i32_b(short128 src, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_convert_i16_to_i32_vb(short128 src, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_convert_i16_to_u32_b(short128 src, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_convert_i16_to_u32_vb(short128 src, int switches, uint64 income, bool64 predicate, bool polarity=0);
float64 v_convert_i8_to_f32_b(char256 src, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_convert_i8_to_f32_vb(char256 src, int switches, float64 income, bool64 predicate, bool polarity=0);
int64 v_convert_i8_to_i32_b(char256 src, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_convert_i8_to_i32_vb(char256 src, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_convert_i8_to_u32_b(char256 src, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_convert_i8_to_u32_vb(char256 src, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_convert_i8_to_i16_b(char256 src, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_convert_i8_to_i16_vb(char256 src, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_convert_i8_to_u16_b(char256 src, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_convert_i8_to_u16_vb(char256 src, int switches, ushort128 income, bool128 predicate, bool polarity=0);
/// @}


/// @brief Represents CONVERT instruction in vector slot in which result vector \n
/// element is shorter then the source one (down-convert).
///
/// @param src       The value to convert (SRC1).
/// @param lane      The lane in output vector to which result of conversion is written, (0: all even lanes, 1: all odd lanes). \n
/// For f32 to i16, only lanes 0-1 are available. For f32 to i8, all lanes 0-3 are available.
/// @param switches  Switches of the instructions.
/// @param income    Income value of DEST.
/// @param predicate Predicate value for the instruction.
/// @param polarity  True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
#if defined(__gaudi_plus__)
///         \li SW_SR - Stochastic Rounding.
///         \li SW_RHAZ - Round half away from zero.
#endif
///         \li SW_CSR - Take rounding mode from
#if defined(__gaudi__) || defined(__goya__)
///                    ROUND_CSR register
#endif
///
/// @{
short128 v_convert_f32_to_i16_b(float64 src, const int lane, int switches, short128 income, bool predicate=1, bool polarity=0);
short128 v_convert_f32_to_i16_vb(float64 src, const int lane, int switches, short128 income, bool128 predicate, bool polarity=0);
char256 v_convert_f32_to_i8_b(float64 src, const int lane, int switches, char256 income, bool predicate=1, bool polarity=0);
char256 v_convert_f32_to_i8_vb(float64 src, const int lane, int switches, char256 income, bool256 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_convert_i32_to_bf16_b(int64 src, const int lane, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
bfloat128 v_convert_i32_to_bf16_vb(int64 src, const int lane, int switches, bfloat128 income, bool64 predicate, bool polarity=0);
#endif
#if defined(__goya__)
uchar256 v_convert_i32_to_u8_b(int64 src, const int lane, int switches, uchar256 income, bool predicate=1, bool polarity=0);
uchar256 v_convert_i32_to_u8_vb(int64 src, const int lane, int switches, uchar256 income, bool64 predicate, bool polarity=0);
#endif

uchar256 v_convert_i16_to_u8_b(short128 src, const int lane, int switches, uchar256 income, bool predicate=1, bool polarity=0);
uchar256 v_convert_i16_to_u8_vb(short128 src, const int lane, int switches, uchar256 income, bool128 predicate, bool polarity=0);
char256 v_convert_i16_to_i8_b(short128 src, const int lane, int switches, char256 income, bool predicate=1, bool polarity=0);
char256 v_convert_i16_to_i8_vb(short128 src, const int lane, int switches, char256 income, bool128 predicate, bool polarity=0);
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents CONVERT instruction in vector slot that make move 1xFP32
/// to 1xBF16/F16 (lane 0 only), which means the 64 values will be assigned into the
/// 128 elements vector in every other lane. E.g. 0,2,4,6 ... 126.
///
/// @param src       The value to convert (SRC1).
/// @param switches  Switches of the instructions.
/// @param income    Income value of DEST.
/// @param predicate Predicate value for the instruction.
/// @param polarity  True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
///         \li SW_RHAZ - Round half away from zero.
///         \li SW_CSR - Take rounding mode from
#endif
#if defined(__gaudi__)
///                    ROUND_CSR register
#endif
/// @{
#if defined(__gaudi_plus__)
bfloat128 v_convert_f32_to_bf16_single_b(float64 src, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
bfloat128 v_convert_f32_to_bf16_single_vb(float64 src, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif

/// @}

#if defined(__gaudi_plus__)
/// @brief Represents CONVERT instruction in vector slot with ALL_LANES switch, \n
/// converts all lanes and vector element size of source and destination is different.
///
/// @param src       The value to convert (SRC1).
/// @param switches  Switches of the instructions.
/// @param income    Income value of DEST.
/// @param predicate Predicate value for the instruction.
/// @param polarity  True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
///         \li SW_RHAZ - Round half away from zero.
///         \li SW_CSR - Take rounding mode from
#endif
#if defined(__gaudi__)
///                    ROUND_CSR register
#endif
/// @{
#if defined(__gaudi_plus__)
bfloat128 v_convert_f32_to_bf16_all_b(float128 src, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
float128 v_convert_bf16_to_f32_all_b(bfloat128 src, int switches=0, float128 income={}, bool predicate=1, bool polarity=0);
float128 v_convert_bf16_to_f32_all_vb(bfloat128 src, int switches, float128 income, bool128 predicate, bool polarity=0);
#endif


/// @}


//
// ------ CONVERT_INT32
//

/// @brief Represents scalar CONVERT_INT32 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_INT32 (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
#if defined(__gaudi_plus__)
///         \li SW_RZ - Round zero.
#endif
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
///
/// @{

// Although the second argument (shift) is declared as INT8 in Goya and Gaudi,
// we use INT32 for all architectures to have the single function.
int16_t s_convert_int32_to_i16(int32_t value, int32_t shift, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
int8_t  s_convert_int32_to_i8 (int32_t value, int32_t shift, int switches=0, int8_t  income={}, bool predicate=1, bool polarity=0);
/// @}


#if defined(__goya__) || defined(__gaudi__)
/// @brief Represents vector CONVERT_INT32 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_INT32 (SRC2).
/// @param lane        Lane number in the output vector, (0: all even lanes, 1: all odd lanes). \n
/// For i32 to i16, only lanes 0-1 are available. For i32 to i8, all lanes 0-3 are available.
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
#endif
#if defined(__gaudi__)
///         \li SW_RZ - Round zero.
///
#endif
/// @{

// These functions are defined for Goya and Gaudi only.
#if defined(__dali__) || defined(__gaudi__)
short128 v_convert_int32_to_i16_b (int64 value, char256 shift, const int lane, int switches, short128 income, bool   predicate=1, bool polarity=0);
short128 v_convert_int32_to_i16_vb(int64 value, char256 shift, const int lane, int switches, short128 income, bool64 predicate, bool polarity=0);
char256  v_convert_int32_to_i8_b  (int64 value, char256 shift, const int lane, int switches, char256  income, bool   predicate=1, bool polarity=0);
char256  v_convert_int32_to_i8_vb (int64 value, char256 shift, const int lane, int switches, char256  income, bool64 predicate, bool polarity=0);
#endif
/// @}


//
// ------ CONVERT_UINT32
//


/// @brief Represents scalar CONVERT_UINT32 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_UINT32 (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
#if defined(__gaudi_plus__)
///         \li SW_RZ - Round zero.
#endif
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
///
/// @{

// Although the second argument (shift) is declared as INT8 in Goya and Gaudi,
// we use INT32 for all architectures to have the single function for all architectures.
uint16_t s_convert_uint32_to_u16(uint32_t value, int32_t shift, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
uint8_t  s_convert_uint32_to_u8 (uint32_t value, int32_t shift, int switches=0, uint8_t  income={}, bool predicate=1, bool polarity=0);
/// @}


#if defined(__goya__) || defined(__gaudi__)
/// @brief Represents vector CONVERT_UINT32 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_UINT32 (SRC2).
/// @param lane        Lane number in the output vector, (0: all even lanes, 1: all odd lanes). \n
/// For u32 to u16, only lanes 0-1 are available. For u32 to u8, all lanes 0-3 are available.
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
#endif
#if defined(__gaudi__)
///         \li SW_RZ - Round zero.
///
#endif
/// @{

// These functions are defined for Goya and Gaudi only.
#if defined(__dali__) || defined(__gaudi__)
ushort128 v_convert_uint32_to_u16_b (uint64 value, char256 shift, const int lane, int switches, ushort128 income, bool predicate=1, bool polarity=0);
ushort128 v_convert_uint32_to_u16_vb(uint64 value, char256 shift, const int lane, int switches, ushort128 income, bool64 predicate, bool polarity=0);
uchar256  v_convert_uint32_to_u8_b  (uint64 value, char256 shift, const int lane, int switches, uchar256  income, bool predicate=1, bool polarity=0);
uchar256  v_convert_uint32_to_u8_vb (uint64 value, char256 shift, const int lane, int switches, uchar256  income, bool64 predicate, bool polarity=0);
#endif
/// @}


//
// ------ CONVERT_INT16
//


/// @brief Represents scalar CONVERT_INT16 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_INT16 (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
#if defined(__gaudi_plus__)
///         \li SW_RZ - Round zero.
#endif
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
///
/// @{

// Although the second argument (shift) is declared as INT8 in Goya and Gaudi,
// we use INT32 for all architectures to have the single function for all architectures.
int8_t s_convert_int16_to_i8(int16_t value, int32_t shift, int switches, int8_t income, bool predicate, bool polarity);
/// @}


#if defined(__goya__) || defined(__gaudi__)
/// @brief Represents vector CONVERT_INT16 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_INT16 (SRC2).
/// @param lane        Lane number in the output vector, (0: all even lanes, 1: all odd lanes). \n
/// For i16 to i8, only lanes 0-1 are available.
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
#endif
#if defined(__gaudi__)
///         \li SW_RZ - Round zero.
///
#endif
/// @{

// These functions are defined for Goya and Gaudi only.
#if defined(__dali__) || defined(__gaudi__)
char256 v_convert_int16_to_i8_b (short128 value, char256 shift, const int lane, int switches, char256 income, bool    predicate, bool polarity);
char256 v_convert_int16_to_i8_vb(short128 value, char256 shift, const int lane, int switches, char256 income, bool128 predicate, bool polarity);
#endif
/// @}


//
// ------ CONVERT_UINT16
//


/// @brief Represents scalar CONVERT_UINT16 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_UINT16 (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
#if defined(__gaudi_plus__)
///         \li SW_RZ - Round zero.
#endif
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
///
/// @{

// Although the second argument (shift) is declared as INT8 in Goya and Gaudi,
// we use INT32 for all architectures to have the single function for all architectures.
uint8_t s_convert_uint16_to_u8(uint16_t value, int32_t shift, int switches, uint8_t income, bool predicate, bool polarity);
/// @}


#if defined(__goya__) || defined(__gaudi__)
/// @brief Represents vector CONVERT_UINT16 instruction.
///
/// @param value       The value to convert (SRC1).
/// @param shift       The shift argument to CONVERT_UINT16 (SRC2).
/// @param lane        Lane number in the output vector, (0:all even lanes, 1:all odd lanes). \n
/// For u16 to u8, only lanes 0-1 are available.
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Converted value.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///         \li SW_SR - Stochastic Rounding.
#endif
#if defined(__gaudi__)
///         \li SW_RZ - Round zero.
///
#endif
/// @{

// These functions are defined for Goya and Gaudi only.
#if defined(__dali__) || defined(__gaudi__)
uchar256 v_convert_uint16_to_u8_b (ushort128 value, char256 shift, const int lane, int switches, uchar256 income, bool    predicate, bool polarity);
uchar256 v_convert_uint16_to_u8_vb(ushort128 value, char256 shift, const int lane, int switches, uchar256 income, bool128 predicate, bool polarity);
#endif
/// @}


//
// ------ EXTRACT_EXP
//


/// @brief Represents EXTRACT_EXP instruction.
///
/// @param a           Input float number (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Mantissa of the input value.
///
/// \par Allowed switches are:
///     \li SW_BIASED - When set, the exponent should be biased (unsigned). When cleared, the exponent should be non-biased (signed).
///
/// @{
int32_t s_f32_extract_exp(float a, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
int16_t s_bf16_extract_exp(bf16 a, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_f32_extract_exp_vb(float64 a, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_f32_extract_exp_b(float64 a, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
short128 v_bf16_extract_exp_vb(bfloat128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_bf16_extract_exp_b(bfloat128 a, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ FCLASS
//

#if defined(__gaudi_plus__)
/// @brief Represents FCLASS instruction.
///
/// @param a           The input value (SRC1).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return 8/10bit-masks format describing the input value.
///
/// 10bit mask maps to the following classes: (returned for 16/32 bit datatypes) \n
/// | bit   | 0    | 1       | 2         | 3  | 4  | 5         | 6       | 7    | 8    | 9    |
/// |-------|------|---------|-----------|----|----|-----------|---------|------|------|------|
/// | class | -Inf | -normal | -denormal | -0 | +0 | +denormal | +normal | +Inf | sNaN | qNaN |
///
/// 8bit mask maps to the following classes: (returned for 8 bit datatypes) \n
/// | bit   | 0    | 1       | 2         | 3        | 4         | 5       | 6    | 7            |
/// |-------|------|---------|-----------|----------|-----------|---------|------|--------------|
/// | class | -Inf | -normal | -denormal | -0 or +0 | +denormal | +normal | +Inf | sNaN or qNaN |
///
#endif
/// @{
#if defined(__gaudi_plus__)
float s_f32_fclass(float a, int switches=0, float income={}, bool predicate=1, bool polarity=0);
bf16 s_bf16_fclass(bf16 a, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
float64 v_f32_fclass_b(float64 a, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_fclass_vb(float64 a, int switches, float64 income, bool64 predicate, bool polarity=0);
bfloat128 v_bf16_fclass_b(bfloat128 a, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_fclass_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
/// @}


/// @brief Represents FIND_FIRST instruction.
///
/// @param a           Input value (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return First 0/1 bit position.
///
/// \par Allowed switches are:
///     - [SET] - Indicates whether the first bit to look for is 0 or 1.
///         \li SW_FIND_ZERO
///         \li SW_FIND_ONE
///     - [DIR] - Indicates the direction of the search.
///         \li SW_LSB
///         \li SW_MSB
///
/// @{
uint8_t s_f32_find_first(float a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
uint8_t s_bf16_find_first(bf16 a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#endif
uint8_t s_i32_find_first(int32_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u32_find_first(uint32_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_i16_find_first(int16_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u16_find_first(uint16_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_i8_find_first(int8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_find_first(uint8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uchar256 v_f32_find_first_b(float64 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_f32_find_first_vb(float64 a, int switches, uchar256 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
uchar256 v_bf16_find_first_b(bfloat128 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_bf16_find_first_vb(bfloat128 a, int switches, uchar256 income, bool128 predicate, bool polarity=0);
#endif
uchar256 v_i32_find_first_b(int64 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_i32_find_first_vb(int64 a, int switches, uchar256 income, bool64 predicate, bool polarity=0);
uchar256 v_u32_find_first_b(uint64 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u32_find_first_vb(uint64 a, int switches, uchar256 income, bool64 predicate, bool polarity=0);
uchar256 v_i16_find_first_b(short128 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_i16_find_first_vb(short128 a, int switches, uchar256 income, bool128 predicate, bool polarity=0);
uchar256 v_u16_find_first_b(ushort128 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u16_find_first_vb(ushort128 a, int switches, uchar256 income, bool128 predicate, bool polarity=0);
uchar256 v_i8_find_first_b(char256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_i8_find_first_vb(char256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_find_first_b(uchar256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_find_first_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


//
// ------ FORM_FP_NUMBER
//


/// @brief Represents FORM_FP_NUMBER instruction.
///
/// @param a           Exponent (SRC1).
/// @param b           Sign (SRC2).
/// @param c           Mantissa (SRC3).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Constructed floating number.
///
/// \par Allowed switches are:
///     \li SW_ADD_BIAS - Add bias to exponent (SRC1) prior to operation.
///     \li SW_FORCE_SIGN0 - Force sign bit at DEST to 0.
///     \li SW_FORCE_SIGN1 - Force sign bit at DEST to 1.
///     \li SW_EXP_IS_NUM - Treat the value in the source as an INT8 number, shift left by mantissa size and mask un-relevant bits.
///     \li SW_SIGN_LSB - When set, takes the SIGN from the LSB of the src instead from its sign bit.
///
/// SW_FORCE_SIGN0, SW_FORCE_SIGN1, SW_SIGN_LSB are mutually exclusive. \n
/// @note In intrinsic functions with '_ie_' infix, SW_EXP_IS_NUM switch is set by default.
///
/// @{
float64 v_f32_form_fp_num_b(float64 a, float64 b, float64 c, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_form_fp_num_vb(float64 a, float64 b, float64 c, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_form_fp_num_ie_b(char256 a, float64 b, float64 c, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_form_fp_num_ie_vb(char256 a, float64 b, float64 c, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_form_fp_num_b(bfloat128 a, bfloat128 b, bfloat128 c, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_form_fp_num_vb(bfloat128 a, bfloat128 b, bfloat128 c, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_form_fp_num_ie_b(char256 a, bfloat128 b, bfloat128 c, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_form_fp_num_ie_vb(char256 a, bfloat128 b, bfloat128 c, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
/// @}


//
// ------ GEN_ADDR
//


/// @brief Represents GEN_ADDR instruction.
///
/// @param inx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Pointer in global address space, pointing to the selected tensor element.
///
/// @{
__global void *gen_addr(int5 inx, const int8_t tensor, int switches=0, __global void *income={}, bool predicate=1, bool polarity=0);
/// @}


//
// ------ GET_LUT_ENTRY_AND_INTERVAL_START
//


/// @brief Represents GET_LUT_ENTRY_AND_INTERVAL_START instruction.
///
/// @param src         Input value (SRC1).
/// @param shift       Significand shift
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Pair of the interval and its offset for the special function approximation.
///
/// \par Allowed switches are:
///     - [FUNC_VARIANT] - Determines the variant of the function, when no switch is set -
///                        all functions other then tanh/sqrt/rsqrt/sin/cos.
///         \li SW_LUT_TANH
///         \li SW_LUT_SQRT_RSQRT
///         \li SW_LUT_SIN_COS
///         \li SW_LUT_LOG
///
/// @{
uint64_float64 v_f32_get_lut_entry_and_interval_start_b(float64 src, const int8_t shift, int switches=0, uint64_float64 income={}, bool predicate=1, bool polarity=0);
uint64_float64 v_f32_get_lut_entry_and_interval_start_vb(float64 src, const int8_t shift, int switches, uint64_float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
ushort128_bfloat128 v_bf16_get_lut_entry_and_interval_start_b(bfloat128 src, const int8_t shift, int switches=0, ushort128_bfloat128 income={}, bool predicate=1, bool polarity=0);
ushort128_bfloat128 v_bf16_get_lut_entry_and_interval_start_vb(bfloat128 src, const int8_t shift, int switches, ushort128_bfloat128 income, bool128 predicate, bool polarity=0);
#endif
/// @}


//
// ------ LD_L
//


/// @brief Represents LD_L instruction.
///
/// @param addr        Address to read from (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
/// \par Allowed switches are:
///     \li SW_MMIO - When set load from MMIO, else - load from SLM.
///
/// @{
float s_f32_ld_l(uint32_t addr, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_ld_l(uint32_t addr, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_ld_l(uint32_t addr, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_ld_l(uint32_t addr, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_ld_l(uint32_t addr, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_ld_l(uint32_t addr, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_ld_l(uint32_t addr, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_ld_l(uint32_t addr, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_ld_l(uint32_t addr, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
/// @}


//
// ------ LD_L_V
//


/// @brief Represents LD_L_V instruction.
///
/// @param addr        Address to read from (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
///
/// @{
float64 v_f32_ld_l_v_b(uint32_t addr, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_l_v_vb(uint32_t addr, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_ld_l_v_b(uint32_t addr, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_ld_l_v_b(uint32_t addr, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_l_v_b(uint32_t addr, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_l_v_b(uint32_t addr, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_l_v_b(uint32_t addr, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_l_v_b(uint32_t addr, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_l_v_b(uint32_t addr, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_l_v_b(uint32_t addr, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
float64 v_f32_ld_l_v_vb(uint32_t addr, int switches, float64 income, bool64 predicate, bool polarity=0);
int64 v_i32_ld_l_v_vb(uint32_t addr, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_l_v_vb(uint32_t addr, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_ld_l_v_vb(uint32_t addr, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_l_v_vb(uint32_t addr, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_ld_l_v_vb(uint32_t addr, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_l_v_vb(uint32_t addr, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_l_v_vb(uint32_t addr, int switches, bool256 income, bool256 predicate, bool polarity=0);
#endif
/// @}


//
// ------ LD_L_V_HIGH
//


/// @brief Represents LD_L_V_HIGH instruction.
///
/// @param addr        Address to read from (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
///
/// @{
float64 v_f32_ld_l_v_high_vb(uint32_t addr, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_ld_l_v_high_b(uint32_t addr, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_l_v_high_vb(uint32_t addr, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_ld_l_v_high_b(uint32_t addr, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_ld_l_v_high_vb(uint32_t addr, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_ld_l_v_high_b(uint32_t addr, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_l_v_high_vb(uint32_t addr, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_l_v_high_b(uint32_t addr, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_l_v_high_vb(uint32_t addr, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_ld_l_v_high_b(uint32_t addr, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_l_v_high_vb(uint32_t addr, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_l_v_high_b(uint32_t addr, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_l_v_high_vb(uint32_t addr, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_ld_l_v_high_b(uint32_t addr, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_l_v_high_vb(uint32_t addr, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_l_v_high_b(uint32_t addr, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_l_v_high_vb(uint32_t addr, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_l_v_high_b(uint32_t addr, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}


//
// ------ LD_L_V_LOW
//


/// @brief Represents LD_L_V_LOW instruction.
///
/// @param addr        Address to read from (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
///
/// @{
float64 v_f32_ld_l_v_low_vb(uint32_t addr, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_ld_l_v_low_b(uint32_t addr, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_l_v_low_vb(uint32_t addr, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_ld_l_v_low_b(uint32_t addr, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_ld_l_v_low_vb(uint32_t addr, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_ld_l_v_low_b(uint32_t addr, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_l_v_low_vb(uint32_t addr, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_l_v_low_b(uint32_t addr, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_l_v_low_vb(uint32_t addr, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_ld_l_v_low_b(uint32_t addr, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_l_v_low_vb(uint32_t addr, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_l_v_low_b(uint32_t addr, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_l_v_low_vb(uint32_t addr, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_ld_l_v_low_b(uint32_t addr, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_l_v_low_vb(uint32_t addr, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_l_v_low_b(uint32_t addr, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_l_v_low_vb(uint32_t addr, int switches, bool256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_l_v_low_b(uint32_t addr, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}


//
// ------ LD_G
//

/// @brief Represents LD_G instruction.
///
/// @param addr        Address to read from (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
///
/// @{
float s_f32_ld_g(__global void *addr, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_ld_g(__global void *addr, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_ld_g(__global void *addr, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_ld_g(__global void *addr, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_ld_g(__global void *addr, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_ld_g(__global void *addr, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_ld_g(__global void *addr, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_ld_g(__global void *addr, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_ld_g(__global void *addr, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_g(__global void *addr, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
float64 v_f32_ld_g(__global void *addr, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ld_g(__global void *addr, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_g(__global void *addr, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_g(__global void *addr, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_g(__global void *addr, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_g(__global void *addr, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_g(__global void *addr, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
/// @}


//
// ------ LD_TNSR
//

/// @brief Represents LD_TNSR instruction.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
/// @{
float64 v_f32_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_tnsr_b(int5 ndx, const int8_t tensor, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
float64 v_f32_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, float64 income, bool64 predicate, bool polarity=0);
bfloat128 v_bf16_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
int64 v_i32_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_tnsr_vb(int5 ndx, int8_t tensor, int switches, bool256 income, bool256 predicate, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_tnsr_b(int5 ndx, int8_t tensor, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents LD_TNSR instruction with PARTIAL switch.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param size        Size in elements minus 1.
/// @param offset      Offset in elements.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
#endif
/// @{
#if defined(__gaudi_plus__)
float64 v_f32_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_tnsr_partial_b(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, float64 income, bool64 predicate, bool polarity=0);
bfloat128 v_bf16_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
int64 v_i32_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_tnsr_partial_vb(int5 ndx, int8_t tensor, int8_t size, int8_t offset, int switches, bool256 income, bool256 predicate, bool polarity=0);
#endif
/// @}


//
// ------ LD_TNSR_HIGH
//

/// @brief Represents LD_TNSR_HIGH instruction.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
///
/// @{
float64 v_f32_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_tnsr_high_b(int5 ndx, const int8_t tensor, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
float64 v_f32_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, float64 income, bool64 predicate, bool polarity=0);
bfloat128 v_bf16_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
int64 v_i32_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_tnsr_high_vb(int5 ndx, int8_t tensor, int switches, bool256 income, bool256 predicate, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_tnsr_high_b(int5 ndx, int8_t tensor, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ LD_TNSR_LOW
//

/// @brief Represents LD_TNSR_LOW instruction.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The loaded value.
///
///
/// @{
float64 v_f32_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_ld_tnsr_low_b(int5 ndx, const int8_t tensor, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
float64 v_f32_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, float64 income, bool64 predicate, bool polarity=0);
bfloat128 v_bf16_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
int64 v_i32_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_ld_tnsr_low_vb(int5 ndx, int8_t tensor, int switches, bool256 income, bool256 predicate, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
bfloat128 v_bf16_ld_tnsr_low_b(int5 ndx, int8_t tensor, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}

//
// ------ MAC
//

/// @brief Represents MAC instruction.
///
/// @param a           The first SRC operand to MAC (SRC1).
/// @param b           The second SRC operand to MAC (SRC2).
/// @param accumulator DEST operand to MAC.
/// @param switches    Switches of MAC instructions.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Updated value of accumulator \c acc=acc+(a*b).
///
/// \par Allowed switches are:
///     \li SW_SAT - Saturates the result (valid for integer data-types only).
///     \li SW_NEG - When set, \c acc=acc-(a*b)
#if defined(__gaudi__) || defined(__goya__)
///                 (valid for float data-types only).
#endif
///
/// @{
float s_f32_mac(float a, float b, float accumulator, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_mac(bf16 a, bf16 b, bf16 accumulator, int switches=0, bool predicate=1, bool polarity=0);
#endif
int32_t s_i16_mac(int16_t a, int16_t b, int32_t accumulator, int switches=0, bool predicate=1, bool polarity=0);
uint32_t s_u16_mac(uint16_t a, uint16_t b, uint32_t accumulator, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__goya__)
int32_t s_i8_mac(int8_t a, int8_t b, int32_t accumulator, int switches=0, bool predicate=1, bool polarity=0);
uint32_t s_u8_mac(uint8_t a, uint8_t b, uint32_t accumulator, int switches=0, bool predicate=1, bool polarity=0);
#endif
float64 v_f32_mac_vb(float64 a, float64 b, float64 accumulator, int switches, bool64 predicate, bool polarity=0);
float64 v_f32_mac_b(float64 a, float64 b, float64 accumulator, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_mac_vb(bfloat128 a, bfloat128 b, bfloat128 accumulator, int switches, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_mac_b(bfloat128 a, bfloat128 b, bfloat128 accumulator, int switches=0, bool predicate=1, bool polarity=0);
#endif
int128 v_i16_mac_vb(short128 a, short128 b, int128 accumulator, int switches, bool128 predicate, bool polarity=0);
int128 v_i16_mac_b(short128 a, short128 b, int128 accumulator, int switches=0, bool predicate=1, bool polarity=0);
uint128 v_u16_mac_vb(ushort128 a, ushort128 b, uint128 accumulator, int switches, bool128 predicate, bool polarity=0);
uint128 v_u16_mac_b(ushort128 a, ushort128 b, uint128 accumulator, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__goya__)
int256 v_i8_mac_vb(char256 a, char256 b, int256 accumulator, int switches, bool256 predicate, bool polarity=0);
int256 v_i8_mac_b(char256 a, char256 b, int256 accumulator, int switches=0, bool predicate=1, bool polarity=0);
uint256 v_u8_mac_vb(uchar256 a, uchar256 b, uint256 accumulator, int switches, bool256 predicate, bool polarity=0);
uint256 v_u8_mac_b(uchar256 a, uchar256 b, uint256 accumulator, int switches=0, bool predicate=1, bool polarity=0);
#endif
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents MAC instruction with ACC_F32 switch (32-bit accumulator).
///
/// @param a           The first SRC operand to MAC (SRC1).
/// @param b           The second SRC operand to MAC (SRC2).
/// @param accumulator DEST operand to MAC.
/// @param switches    Switches of MAC instructions.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Updated value of accumulator \c acc=acc+(a*b).
///
/// \par Allowed switches are:
///     \li SW_NEG - When set, \c acc=acc-(a*b)
///
#endif
/// @{
#if defined(__gaudi_plus__)
float s_bf16_mac_acc32(bf16 a, bf16 b, float accumulator, int switches=0, bool predicate=1, bool polarity=0);
#endif
#if defined(__gaudi_plus__)
float128 v_bf16_mac_acc32_vb(bfloat128 a, bfloat128 b, float128 accumulator, int switches, bool128 predicate, bool polarity=0);
float128 v_bf16_mac_acc32_b(bfloat128 a, bfloat128 b, float128 accumulator, int switches=0, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ MAX
//

/// @brief Represents MAX instruction.
///
/// @param a           The first SRC operand to MAX (SRC1).
/// @param b           The second SRC operand to MAX (SRC2).
/// @param switches    Switches of MAX instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_max(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_max(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_max(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_max(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_max(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_max(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_max(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_max(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
float64 v_f32_max_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_max_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_max_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_max_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_max_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_max_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_max_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_max_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_max_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_max_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_max_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_max_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_max_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_max_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_max_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_max_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
/// @}


/// @brief Represents scalar MAX instruction for integer operands in index registers.
///
/// @param a           The first SRC operand to MAX (SRC1).
/// @param b           The second SRC operand to MAX (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of MAX instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int5 i_i32_max(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ MIN
//

/// @brief Represents MIN instruction.
///
/// @param a           The first SRC operand to MIN (SRC1).
/// @param b           The second SRC operand to MIN (SRC2).
/// @param switches    Switches of MIN instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_min(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_min(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_min(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_min(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_min(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_min(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_min(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_min(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
float64 v_f32_min_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_min_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_min_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_min_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_min_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_min_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_min_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_min_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_min_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_min_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_min_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_min_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_min_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_min_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_min_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_min_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
/// @}


/// @brief Represents scalar MIN instruction for integer operands in index registers.
///
/// @param a           The first SRC operand to MIN (SRC1).
/// @param b           The second SRC operand to MIN (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of MIN instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int5 i_i32_min(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}

//
// ------ MOV
//

/// @brief Represents MOV instruction.
///
/// @param a           Source (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation - move \p a to DEST.
///
/// @{
float s_f32_mov(float a, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_mov(bf16 a, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_mov(int32_t a, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_mov(uint32_t a, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_mov(int16_t a, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_mov(uint16_t a, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_mov(int8_t a, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_mov(uint8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_mov(bool a, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
float64 v_f32_mov_vb(float64 a, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_mov_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
int64 v_i32_mov_vb(int64 a, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_mov_vb(uint64 a, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_mov_vb(short128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_mov_vb(ushort128 a, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_mov_vb(char256 a, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_mov_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_mov_vb(bool256 a, int switches, bool256 income, bool256 predicate, bool polarity=0);
float64 v_f32_mov_b(float64 a, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_mov_b(bfloat128 a, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_mov_b(int64 a, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_mov_b(uint64 a, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_mov_b(short128 a, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_mov_b(ushort128 a, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_mov_b(char256 a, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_mov_b(uchar256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_mov_b(bool256 a, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
/// @}


/// @brief Represents MOV instruction for int5 type.
///
/// @param a           Source (SRC1).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation - move \p a to DEST.
///
/// @{
int5 i_i32_mov(int5 a, int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


/// @brief Represents MOV instruction with argument FLAVOR.
///
/// @param a           Source (SRC1).
/// @param flavor      Selects section of VPRF participated in the operation [0-8]. \n
///                    0-7 map to 8 VPRF sections, when \p flavor is set to 8 - mov to all 8 sections.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation - move \p a to one or all 8 sections of DEST.
///
/// @{
bool256 v_i1_mov_flavor_b(int32_t a, const int flavor, int switches, bool256 income, bool predicate=1, bool polarity=0);
bool256 v_i1_mov_flavor_vb(int32_t a, const int flavor, int switches, bool256 income, bool256 predicate, bool polarity=0);
/// @}


/// @brief Represent MOV SPRF to VPRF instruction.
///
/// @param a           Source (SRC1).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation - broadcast \p a to DEST.
///
/// @{
bool256 v_i1_mov_i1_b(bool a, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_mov_i1_vb(bool a, int switches, bool256 income, bool256 predicate, bool polarity=0);
/// @}


/// @brief Represents MOV_DUAL_GROUP instruction.
///
/// @param a           Source (SRC1).
/// @param b           Byte write mask. 
/// @param src_dg      Source dual group {0, 1, 2, 3}.
/// @param dest_dg     Destination dual group {0, 1, 2, 3}.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// \par Allowed switches:
///     \li SW_WR_LOWER_GROUP
///     \li SW_WR_UPPER_GROUP
///
/// @{
float64 v_f32_mov_dual_group_vb(float64 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_mov_dual_group_b(float64 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, float64 income, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_mov_dual_group_vb(bfloat128 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_mov_dual_group_b(bfloat128 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_mov_dual_group_vb(int64 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_mov_dual_group_b(int64 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, int64 income, bool predicate=1, bool polarity=0);
uint64 v_u32_mov_dual_group_vb(uint64 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_mov_dual_group_b(uint64 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, uint64 income, bool predicate=1, bool polarity=0);
short128 v_i16_mov_dual_group_vb(short128 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_mov_dual_group_b(short128 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, short128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_mov_dual_group_vb(ushort128 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_mov_dual_group_b(ushort128 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, ushort128 income, bool predicate=1, bool polarity=0);
char256 v_i8_mov_dual_group_vb(char256 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_mov_dual_group_b(char256 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, char256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_mov_dual_group_vb(uchar256 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_mov_dual_group_b(uchar256 a, const uint32_t b, const int src_dg, const int dest_dg, int switches, uchar256 income, bool predicate=1, bool polarity=0);
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents MOV_DUAL_GROUP instruction with ALL switch.
///
/// @param a           Source (SRC1).
/// @param b           Byte write mask.
/// @param sdg0        Source dual group 0 {0, 1, 2, 3}.
/// @param sdg1        Source dual group 1 {0, 1, 2, 3}.
/// @param sdg2        Source dual group 2 {0, 1, 2, 3}.
/// @param sdg3        Source dual group 3 {0, 1, 2, 3}.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// \par Allowed switches:
///     \li SW_WR_LOWER_GROUP0
///     \li SW_WR_LOWER_GROUP1
///     \li SW_WR_LOWER_GROUP2
///     \li SW_WR_LOWER_GROUP3
///     \li SW_WR_UPPER_GROUP0
///     \li SW_WR_UPPER_GROUP1
///     \li SW_WR_UPPER_GROUP2
///     \li SW_WR_UPPER_GROUP3
///
#endif
/// @{
#if defined(__gaudi_plus__)
float64 v_f32_mov_dual_group_all_vb(float64 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_mov_dual_group_all_b(float64 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, float64 income, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_mov_dual_group_all_vb(bfloat128 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_mov_dual_group_all_b(bfloat128 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
int64 v_i32_mov_dual_group_all_vb(int64 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_mov_dual_group_all_b(int64 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, int64 income, bool predicate=1, bool polarity=0);
uint64 v_u32_mov_dual_group_all_vb(uint64 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_mov_dual_group_all_b(uint64 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, uint64 income, bool predicate=1, bool polarity=0);
short128 v_i16_mov_dual_group_all_vb(short128 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_mov_dual_group_all_b(short128 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, short128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_mov_dual_group_all_vb(ushort128 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_mov_dual_group_all_b(ushort128 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, ushort128 income, bool predicate=1, bool polarity=0);
char256 v_i8_mov_dual_group_all_vb(char256 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_mov_dual_group_all_b(char256 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, char256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_mov_dual_group_all_vb(uchar256 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_mov_dual_group_all_b(uchar256 a, const uint32_t b, const int sdg0, const int sdg1, const int sdg2, const int sdg3, int switches, uchar256 income, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ MOV_IRF_DIM
//


/// @brief Represents MOV_IRF_DIM instruction.
///
/// @param src         The index vector (SRC1).
/// @param dim         The number of extracted dimension.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation - move \p src to DEST.
///
/// @{
int32_t mov_irf_dim(int5 src, const int8_t dim, int switches, int32_t income={}, bool predicate=1, bool polarity=0);
/// @}


/// @brief Represents MOV_GROUP instruction.
///
/// @param a           Source (SRC1).
/// @param b           Byte write mask.
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// \par Allowed switches:
///     - [GROUP_EN] - when set group is written according to \p b.
///         \li SW_GROUP0_EN
///         \li SW_GROUP1_EN
///     - [DUAL_GROUP_EN] - when set dual group is written according to GROUP_EN and \p b.
///         \li SW_DUAL_GROUP0_EN
///         \li SW_DUAL_GROUP1_EN
///         \li SW_DUAL_GROUP2_EN
///         \li SW_DUAL_GROUP3_EN
///
/// @{
float64 v_f32_mov_group_vb(float64 a, uint32_t b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_mov_group_b(float64 a, uint32_t b, int switches, float64 income, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_mov_group_vb(bfloat128 a, uint32_t b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_mov_group_b(bfloat128 a, uint32_t b, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_mov_group_vb(int64 a, uint32_t b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_mov_group_b(int64 a, uint32_t b, int switches, int64 income, bool predicate=1, bool polarity=0);
uint64 v_u32_mov_group_vb(uint64 a, uint32_t b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_mov_group_b(uint64 a, uint32_t b, int switches, uint64 income, bool predicate=1, bool polarity=0);
short128 v_i16_mov_group_vb(short128 a, uint32_t b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_mov_group_b(short128 a, uint32_t b, int switches, short128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_mov_group_vb(ushort128 a, uint32_t b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_mov_group_b(ushort128 a, uint32_t b, int switches, ushort128 income, bool predicate=1, bool polarity=0);
char256 v_i8_mov_group_vb(char256 a, uint32_t b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_mov_group_b(char256 a, uint32_t b, int switches, char256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_mov_group_vb(uchar256 a, uint32_t b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_mov_group_b(uchar256 a, uint32_t b, int switches, uchar256 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ MSAC
//


#if defined(__goya__)
/// @brief Represents MSAC instruction.
///
/// @param a           Source #1 (SRC1)
/// @param b           Source #2 (SRC2)
/// @param n1          ABCin norm factor, range is [-32, 0] (SRC3)
/// @param n2          CinCout norm factor, range is [-25, 0] (SRC4)
/// @param switches    Switches of MSAC instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Normalized result after multiplication of two source operands (a & b).
///
/// \par Allowed switches:
///     \li SW_RHU - Round half up.
/// \par
///     - [SW_NORMALIZE] - \p c refers to DEST
///         \li SW_NORMALIZE_AB - \c c=((a*b)*2^n1+c)*2^n2)
///         \li SW_NORMALIZE_C - \c c=((a*b)+c*2^n1)*2^n2)
///
#endif
/// @{
#if defined(__goya__)
short128 v_i16_msac_b(short128 a, short128 b, char256 n1, char256 n2, int switches, short128 income, bool predicate=1, bool polarity=0);
short128 v_i16_msac_vb(short128 a, short128 b, char256 n1, char256 n2, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_msac_b(ushort128 a, ushort128 b, char256 n1, char256 n2, int switches, ushort128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_msac_vb(ushort128 a, ushort128 b, char256 n1, char256 n2, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_msac_b(char256 a, char256 b, char256 n1, char256 n2, int switches, char256 income, bool predicate=1, bool polarity=0);
char256 v_i8_msac_vb(char256 a, char256 b, char256 n1, char256 n2, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_msac_b(uchar256 a, uchar256 b, char256 n1, char256 n2, int switches, uchar256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_msac_vb(uchar256 a, uchar256 b, char256 n1, char256 n2, int switches, uchar256 income, bool256 predicate, bool polarity=0);
#endif
/// @}


//
// ------ MUL
//


/// @brief Represents MUL instruction for float arguments.
///
/// @param a           The first SRC operand to MUL (SRC1).
/// @param b           The second SRC operand to MUL (SRC2).
/// @param switches    Switches of MUL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_mul(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_mul(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
float64 v_f32_mul_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_mul_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_mul_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_mul_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents MUL instruction with ACC_F32 switch (32-bit accumulator).
///
/// @param a           The first SRC operand to MUL (SRC1).
/// @param b           The second SRC operand to MUL (SRC2).
/// @param switches    Switches of MUL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
#endif
/// @{
#if defined(__gaudi_plus__)
float s_bf16_mul_acc32(bf16 a, bf16 b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
float128 v_bf16_mul_acc32_vb(bfloat128 a, bfloat128 b, int switches, float128 income, bool128 predicate, bool polarity=0);
float128 v_bf16_mul_acc32_b(bfloat128 a, bfloat128 b, int switches=0, float128 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


/// @brief Represents scalar MUL instruction for integer arguments.
///
/// @param a           The first SRC operand to MUL (SRC1).
/// @param b           The second SRC operand to MUL (SRC2).
/// @param switches    Switches of MUL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// \par Allowed switches are:
///     \li SW_UPPER32 - When set upper 32 bit of 64 bit result are stored (INT32, UINT32 only).
///
/// @{
int32_t s_i32_mul(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_mul(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int32_t s_i16_mul(int16_t a, int16_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u16_mul(uint16_t a, uint16_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
#if defined(__goya__)
int32_t s_i8_mul(int8_t a, int8_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u8_mul(uint8_t a, uint8_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


/// @brief Represents vector MUL instruction for integer arguments.
///
/// @param a           The first SRC operand to MUL (SRC1).
/// @param b           The second SRC operand to MUL (SRC2).
/// @param switches    Switches of MUL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
///
/// @{
int128 v_i32_mul_b(int64 a, int64 b, int switches=0, int128 income={}, bool predicate=1, bool polarity=0);
int128 v_i32_mul_vb(int64 a, int64 b, int switches, int128 income, bool64 predicate, bool polarity=0);
uint128 v_u32_mul_b(uint64 a, uint64 b, int switches=0, uint128 income={}, bool predicate=1, bool polarity=0);
uint128 v_u32_mul_vb(uint64 a, uint64 b, int switches, uint128 income, bool64 predicate, bool polarity=0);
int128 v_i16_mul_b(short128 a, short128 b, int switches=0, int128 income={}, bool predicate=1, bool polarity=0);
int128 v_i16_mul_vb(short128 a, short128 b, int switches, int128 income, bool128 predicate, bool polarity=0);
uint128 v_u16_mul_vb(ushort128 a, ushort128 b, int switches, uint128 income, bool128 predicate, bool polarity=0);
uint128 v_u16_mul_b(ushort128 a, ushort128 b, int switches=0, uint128 income={}, bool predicate=1, bool polarity=0);
#if defined(__goya__)
int256 v_i8_mul_vb(char256 a, char256 b, int switches, int256 income, bool256 predicate, bool polarity=0);
int256 v_i8_mul_b(char256 a, char256 b, int switches=0, int256 income={}, bool predicate=1, bool polarity=0);
uint256 v_u8_mul_vb(uchar256 a, uchar256 b, int switches, uint256 income, bool256 predicate, bool polarity=0);
uint256 v_u8_mul_b(uchar256 a, uchar256 b, int switches=0, uint256 income={}, bool predicate=1, bool polarity=0);
#endif
/// @}


/// @brief Represents vector MUL instruction with DOUBLE_AND_ROUND32 switch (RHU).
///
/// @param a           The first SRC operand to MUL (SRC1).
/// @param b           The second SRC operand to MUL (SRC2).
/// @param switches    Switches of MUL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int64 v_i32_mul_round_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_mul_round_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_mul_round_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_mul_round_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
/// @}


/// @brief Represents scalar MUL instruction for integer operands in index registers.
///
/// @param a           The first SRC operand to MUL (SRC1).
/// @param b           The second SRC operand to MUL (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of MUL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int5 i_i32_mul(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ NEARBYINT
//

/// @brief Represents NEARBYINT instruction.
///
/// @param a           Input value (SRC1).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Returns the nearest integer value to \p a.
///
/// \par Allowed switches are:
///     - [ROUND]
///         \li SW_RHNE - Round half to nearest even.
///         \li SW_RZ - Round zero.
///         \li SW_RU - Round up.
///         \li SW_RD - Round down.
///
/// @{
float s_f32_nearbyint(float a, int switches=0, float income={}, bool predicate=1, bool polarity=0);
float64 v_f32_nearbyint_b(float64 a, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_nearbyint_vb(float64 a, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_nearbyint(bf16 a, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_nearbyint_b(bfloat128 a, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_nearbyint_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
/// @}


//
// ------ NOT
//

/// @brief Represents NOT instruction.
///
/// @param a           The argument (SRC1).
/// @param switches    Switches of NOT instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_not(float a, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_not(bf16 a, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_not(int32_t a, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_not(uint32_t a, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_not(int16_t a, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_not(uint16_t a, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_not(int8_t a, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_not(uint8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_not(bool a, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
float64 v_f32_not_b(float64 a, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_not_vb(float64 a, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_not_b(bfloat128 a, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_not_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
int64 v_i32_not_b(int64 a, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_not_vb(int64 a, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_not_b(uint64 a, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_not_vb(uint64 a, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_not_b(short128 a, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_not_vb(short128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_not_b(ushort128 a, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_not_vb(ushort128 a, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_not_b(char256 a, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_not_vb(char256 a, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_not_b(uchar256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_not_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
bool256 v_i1_not_b(bool256 a, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_not_vb(bool256 a, int switches, bool256 income, bool256 predicate, bool polarity=0);
/// @}


/// @brief Represents NOT instruction for int5 operands.
///
/// @param a           The argument (SRC1).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of NOT instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int5 i_i32_not(int5 a, int dimmask, const int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ OR
//

/// @brief Represents OR instruction.
///
/// @param a           The first SRC operand to OR (SRC1).
/// @param b           The second SRC operand to OR (SRC2).
/// @param switches    Switches of OR instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_or(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_or(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_or(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_or(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_or(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_or(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_or(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_or(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_or(bool a, bool b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);
float64 v_f32_or_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_or_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_or_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_or_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_or_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_or_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_or_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_or_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_or_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_or_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_or_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_or_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_or_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_or_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_or_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_or_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_or_b(bool256 a, bool256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_or_vb(bool256 a, bool256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
/// @}


/// @brief Represents OR instruction for int5.
///
/// @param a           The first SRC operand to OR (SRC1).
/// @param b           The second SRC operand to OR (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of OR instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int5 i_i32_or(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}


//
// ------ PACK
//

/// @brief Represents PACK instruction.
///
/// @param a           Value in which population is counted (SRC1).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Packed value.
///
/// \par Allowed switches are:
///     - [GROUP_SOURCE] - Indicates which of the groups in the dual group is to be used as a source.
///         \li SW_GROUP_0
///         \li SW_GROUP_1
///     - [ELEMENT_STRIDE]
///         \li SW_STRIDE_2 - Every second element is valid.
///         \li SW_STRIDE_4 - Every forth element is valid.
///
/// @{
#if defined(__gaudi_plus__)
bfloat128 v_bf16_pack_b(bfloat128 a,int switches, bfloat128 income, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_pack_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
short128 v_i16_pack_b(short128 a,int switches, short128 income, bool predicate=1, bool polarity=0);
short128 v_i16_pack_vb(short128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_pack_b(ushort128 a,int switches, ushort128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_pack_vb(ushort128 a, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_pack_b(char256 a,int switches, char256 income, bool predicate=1, bool polarity=0);
char256 v_i8_pack_vb(char256 a, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_pack_b(uchar256 a,int switches, uchar256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_pack_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


//
// ------ POPCNT
//

/// @brief Represents POPCNT instruction.
///
/// @param a           Value in which population is counted (SRC1).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return The number of 0s or 1s in the elemenet - \p a.
///
/// \par Allowed switches are:
///     - [SET] - Indicates whether to count 0s or 1s.
///         \li SW_COUNT_ZEROS
///         \li SW_COUNT_ONES
///
/// @{
uint8_t s_f32_popcnt(float a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
uint8_t s_bf16_popcnt(bf16 a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
#endif
uint8_t s_i32_popcnt(int32_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u32_popcnt(uint32_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_i16_popcnt(int16_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u16_popcnt(uint16_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_i8_popcnt(int8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_popcnt(uint8_t a, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
uchar256 v_f32_popcnt_b(float64 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_f32_popcnt_vb(float64 a, int switches, uchar256 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
uchar256 v_bf16_popcnt_b(bfloat128 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_bf16_popcnt_vb(bfloat128 a, int switches, uchar256 income, bool128 predicate, bool polarity=0);
#endif
uchar256 v_i32_popcnt_b(int64 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_i32_popcnt_vb(int64 a, int switches, uchar256 income, bool64 predicate, bool polarity=0);
uchar256 v_u32_popcnt_b(uint64 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u32_popcnt_vb(uint64 a, int switches, uchar256 income, bool64 predicate, bool polarity=0);
uchar256 v_i16_popcnt_b(short128 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_i16_popcnt_vb(short128 a, int switches, uchar256 income, bool128 predicate, bool polarity=0);
uchar256 v_u16_popcnt_b(ushort128 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u16_popcnt_vb(ushort128 a, int switches, uchar256 income, bool128 predicate, bool polarity=0);
uchar256 v_i8_popcnt_b(char256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_i8_popcnt_vb(char256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_popcnt_b(uchar256 a, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_popcnt_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


//
// ------ PREFETCH
//

#if defined(__gaudi_plus__)
/// @brief Represents PREFETCH instruction.
///
/// @param a           Address to prefetch (SRC1).
/// @param switches    Switches of the instruction.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
#endif
/// @{
#if defined(__gaudi_plus__)
void prefetch(__global void *a, int switches=0, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ PRMT_INDX
//


/// @brief Represents PRMT_INDX instruction.
///
/// @param ndx         Source tensor coordinates (SRC1).
/// @param prmt_type   Permutation type - each 3 bits (X5) represent a src dimension in \p ndx (SRC2).
/// @param switches    Switches of the instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Tensor coordinates after permutation.
///
/// @{
int5 prmt_indx(int5 ndx, int prmt_type, int switches, int5 income={}, bool predicate=1, bool polarity=0);
/// @}


//
// ------ SEL_EQ
//

/// @brief Represents SEL_EQ instruction.
///
/// @param a           Source #1 to compare (SRC1).
/// @param b           Source #2 to compare (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return One of the sources - \p c or \p d with respect to the comparison of the two other sources - \p a and \p b.
///
#if defined(__gaudi_plus__)
/// \par Allowed switches are:
///     \li SW_MASK_EQ_ZERO - Compare between (a & b) and 0.
#endif
///
/// @{
float64 v_f32_sel_eq_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_eq_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_eq_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_eq_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_eq_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_eq_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat128 v_bf16_sel_eq_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_eq_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_eq_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_eq_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_eq_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_eq_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif


int64 v_i32_sel_eq_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_eq_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_eq_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_eq_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_eq_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_eq_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);

uint64 v_u32_sel_eq_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_eq_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_eq_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_eq_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_eq_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_eq_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128 v_i16_sel_eq_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_eq_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
short128 v_i16_sel_eq_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_eq_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sel_eq_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_eq_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128 v_u16_sel_eq_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_eq_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128 v_u16_sel_eq_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_eq_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sel_eq_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_eq_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);

char256 v_i8_sel_eq_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_eq_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sel_eq_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_eq_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);

uchar256 v_u8_sel_eq_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_eq_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sel_eq_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_eq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL_NEQ
//

/// @brief Represents SEL_NEQ instruction.
///
/// @param a           Source #1 to compare (SRC1).
/// @param b           Source #2 to compare (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return One of the sources - \p c or \p d with respect to the comparison of the two other sources - \p a and \p b.
///
/// @{
float64 v_f32_sel_neq_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_neq_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_neq_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_neq_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_neq_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_neq_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat128 v_bf16_sel_neq_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_neq_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_neq_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_neq_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_neq_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_neq_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif


int64 v_i32_sel_neq_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_neq_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_neq_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_neq_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_neq_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_neq_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);

uint64 v_u32_sel_neq_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_neq_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_neq_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_neq_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_neq_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_neq_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128 v_i16_sel_neq_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_neq_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
short128 v_i16_sel_neq_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_neq_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sel_neq_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_neq_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128 v_u16_sel_neq_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_neq_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128 v_u16_sel_neq_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_neq_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sel_neq_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_neq_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);

char256 v_i8_sel_neq_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_neq_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sel_neq_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_neq_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);

uchar256 v_u8_sel_neq_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_neq_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sel_neq_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_neq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL_LESS
//

/// @brief Represents SEL_LESS instruction.
///
/// @param a           Source #1 to compare (SRC1).
/// @param b           Source #2 to compare (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return One of the sources - \p c or \p d with respect to the comparison of the two other sources - \p a and \p b.
///
/// @{
float64 v_f32_sel_less_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_less_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_less_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_less_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_less_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_less_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat128 v_bf16_sel_less_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_less_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_less_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_less_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_less_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_less_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif


int64 v_i32_sel_less_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_less_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_less_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_less_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_less_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_less_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);

uint64 v_u32_sel_less_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_less_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_less_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_less_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_less_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_less_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128 v_i16_sel_less_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_less_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
short128 v_i16_sel_less_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_less_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sel_less_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_less_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128 v_u16_sel_less_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_less_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128 v_u16_sel_less_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_less_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sel_less_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_less_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);

char256 v_i8_sel_less_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_less_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sel_less_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_less_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);

uchar256 v_u8_sel_less_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_less_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sel_less_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_less_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL_LEQ
//

/// @brief Represents SEL_LEQ instruction.
///
/// @param a           Source #1 to compare (SRC1).
/// @param b           Source #2 to compare (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return One of the sources - \p c or \p d with respect to the comparison of the two other sources - \p a and \p b.
///
/// @{
float64 v_f32_sel_leq_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_leq_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_leq_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_leq_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_leq_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_leq_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat128 v_bf16_sel_leq_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_leq_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_leq_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_leq_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_leq_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_leq_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif


int64 v_i32_sel_leq_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_leq_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_leq_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_leq_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_leq_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_leq_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);

uint64 v_u32_sel_leq_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_leq_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_leq_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_leq_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_leq_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_leq_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128 v_i16_sel_leq_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_leq_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
short128 v_i16_sel_leq_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_leq_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sel_leq_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_leq_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128 v_u16_sel_leq_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_leq_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128 v_u16_sel_leq_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_leq_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sel_leq_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_leq_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);

char256 v_i8_sel_leq_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_leq_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sel_leq_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_leq_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);

uchar256 v_u8_sel_leq_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_leq_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sel_leq_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_leq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL_GRT
//

/// @brief Represents SEL_GRT instruction.
///
/// @param a           Source #1 to compare (SRC1).
/// @param b           Source #2 to compare (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return One of the sources - \p c or \p d with respect to the comparison of the two other sources - \p a and \p b.
///
/// @{
float64 v_f32_sel_grt_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_grt_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_grt_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_grt_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_grt_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_grt_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat128 v_bf16_sel_grt_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_grt_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_grt_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_grt_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_grt_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_grt_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif


int64 v_i32_sel_grt_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_grt_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_grt_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_grt_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_grt_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_grt_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);

uint64 v_u32_sel_grt_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_grt_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_grt_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_grt_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_grt_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_grt_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128 v_i16_sel_grt_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_grt_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
short128 v_i16_sel_grt_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_grt_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sel_grt_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_grt_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128 v_u16_sel_grt_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_grt_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128 v_u16_sel_grt_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_grt_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sel_grt_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_grt_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);

char256 v_i8_sel_grt_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_grt_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sel_grt_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_grt_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);

uchar256 v_u8_sel_grt_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_grt_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sel_grt_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_grt_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL_GEQ
//

/// @brief Represents SEL_GEQ instruction.
///
/// @param a           Source #1 to compare (SRC1).
/// @param b           Source #2 to compare (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return One of the sources - \p c or \p d with respect to the comparison of the two other sources - \p a and \p b.
///
/// @{
float64 v_f32_sel_geq_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_geq_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_geq_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_geq_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_sel_geq_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sel_geq_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat128 v_bf16_sel_geq_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_geq_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_geq_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_geq_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_sel_geq_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sel_geq_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif


int64 v_i32_sel_geq_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_geq_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_geq_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_geq_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_sel_geq_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sel_geq_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);

uint64 v_u32_sel_geq_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_geq_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_geq_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_geq_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sel_geq_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sel_geq_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128 v_i16_sel_geq_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_geq_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
#endif
short128 v_i16_sel_geq_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_geq_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sel_geq_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sel_geq_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128 v_u16_sel_geq_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_geq_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128 v_u16_sel_geq_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_geq_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sel_geq_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sel_geq_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);

char256 v_i8_sel_geq_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_geq_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sel_geq_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sel_geq_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);

uchar256 v_u8_sel_geq_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_geq_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sel_geq_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sel_geq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL2_LESS
//

/// @brief Represents SEL2_LESS instruction.
///
/// @param a           Source #1 to compare and select (SRC1).
/// @param b           Source #2 to compare and select (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Pair (c, a) or (d, b) with respect to the comparison of \p a and \p b.
///
/// @{
float128 v_f32_sel2_less_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float128 income, bool64 predicate, bool polarity=0);
float128 v_f32_sel2_less_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float128 income={}, bool predicate=1, bool polarity=0);
float64_int64 v_f32_sel2_less_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64_int64 income, bool64 predicate, bool polarity=0);
float64_int64 v_f32_sel2_less_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64_int64 income={}, bool predicate=1, bool polarity=0);
float64_uint64 v_f32_sel2_less_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64_uint64 income, bool64 predicate, bool polarity=0);
float64_uint64 v_f32_sel2_less_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64_uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat256 v_bf16_sel2_less_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat256 income={}, bool predicate=1, bool polarity=0);
bfloat256 v_bf16_sel2_less_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat256 income, bool128 predicate, bool polarity=0);
bfloat128_short128 v_bf16_sel2_less_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_short128 income={}, bool predicate=1, bool polarity=0);
bfloat128_short128 v_bf16_sel2_less_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_short128 income, bool128 predicate, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_less_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_ushort128 income={}, bool predicate=1, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_less_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_ushort128 income, bool128 predicate, bool polarity=0);
#endif


int64_float64 v_i32_sel2_less_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64_float64 income, bool64 predicate, bool polarity=0);
int64_float64 v_i32_sel2_less_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64_float64 income={}, bool predicate=1, bool polarity=0);
int128 v_i32_sel2_less_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int128 income, bool64 predicate, bool polarity=0);
int128 v_i32_sel2_less_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int128 income={}, bool predicate=1, bool polarity=0);
int64_uint64 v_i32_sel2_less_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64_uint64 income, bool64 predicate, bool polarity=0);
int64_uint64 v_i32_sel2_less_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64_uint64 income={}, bool predicate=1, bool polarity=0);

uint64_float64 v_u32_sel2_less_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64_float64 income, bool64 predicate, bool polarity=0);
uint64_float64 v_u32_sel2_less_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64_float64 income={}, bool predicate=1, bool polarity=0);
uint64_int64 v_u32_sel2_less_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64_int64 income, bool64 predicate, bool polarity=0);
uint64_int64 v_u32_sel2_less_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64_int64 income={}, bool predicate=1, bool polarity=0);
uint128 v_u32_sel2_less_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint128 income, bool64 predicate, bool polarity=0);
uint128 v_u32_sel2_less_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128_bfloat128 v_i16_sel2_less_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128_bfloat128 income, bool128 predicate, bool polarity=0);
short128_bfloat128 v_i16_sel2_less_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
short256 v_i16_sel2_less_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short256 income, bool128 predicate, bool polarity=0);
short256 v_i16_sel2_less_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short256 income={}, bool predicate=1, bool polarity=0);
short128_ushort128 v_i16_sel2_less_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128_ushort128 income, bool128 predicate, bool polarity=0);
short128_ushort128 v_i16_sel2_less_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128_ushort128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128_bfloat128 v_u16_sel2_less_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128_bfloat128 income, bool128 predicate, bool polarity=0);
ushort128_bfloat128 v_u16_sel2_less_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128_short128 v_u16_sel2_less_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128_short128 income, bool128 predicate, bool polarity=0);
ushort128_short128 v_u16_sel2_less_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128_short128 income={}, bool predicate=1, bool polarity=0);
ushort256 v_u16_sel2_less_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort256 income, bool128 predicate, bool polarity=0);
ushort256 v_u16_sel2_less_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort256 income={}, bool predicate=1, bool polarity=0);

char512 v_i8_sel2_less_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char512 income, bool256 predicate, bool polarity=0);
char512 v_i8_sel2_less_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char512 income={}, bool predicate=1, bool polarity=0);
char256_uchar256 v_i8_sel2_less_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256_uchar256 income, bool256 predicate, bool polarity=0);
char256_uchar256 v_i8_sel2_less_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256_uchar256 income={}, bool predicate=1, bool polarity=0);

uchar256_char256 v_u8_sel2_less_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256_char256 income, bool256 predicate, bool polarity=0);
uchar256_char256 v_u8_sel2_less_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256_char256 income={}, bool predicate=1, bool polarity=0);
uchar512 v_u8_sel2_less_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar512 income, bool256 predicate, bool polarity=0);
uchar512 v_u8_sel2_less_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar512 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL2_LEQ
//

/// @brief Represents SEL2_LEQ instruction.
///
/// @param a           Source #1 to compare and select (SRC1).
/// @param b           Source #2 to compare and select (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Pair (c, a) or (d, b) with respect to the comparison of \p a and \p b.
///
/// @{
float128 v_f32_sel2_leq_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float128 income, bool64 predicate, bool polarity=0);
float128 v_f32_sel2_leq_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float128 income={}, bool predicate=1, bool polarity=0);
float64_int64 v_f32_sel2_leq_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64_int64 income, bool64 predicate, bool polarity=0);
float64_int64 v_f32_sel2_leq_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64_int64 income={}, bool predicate=1, bool polarity=0);
float64_uint64 v_f32_sel2_leq_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64_uint64 income, bool64 predicate, bool polarity=0);
float64_uint64 v_f32_sel2_leq_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64_uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat256 v_bf16_sel2_leq_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat256 income={}, bool predicate=1, bool polarity=0);
bfloat256 v_bf16_sel2_leq_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat256 income, bool128 predicate, bool polarity=0);
bfloat128_short128 v_bf16_sel2_leq_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_short128 income={}, bool predicate=1, bool polarity=0);
bfloat128_short128 v_bf16_sel2_leq_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_short128 income, bool128 predicate, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_leq_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_ushort128 income={}, bool predicate=1, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_leq_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_ushort128 income, bool128 predicate, bool polarity=0);
#endif


int64_float64 v_i32_sel2_leq_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64_float64 income, bool64 predicate, bool polarity=0);
int64_float64 v_i32_sel2_leq_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64_float64 income={}, bool predicate=1, bool polarity=0);
int128 v_i32_sel2_leq_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int128 income, bool64 predicate, bool polarity=0);
int128 v_i32_sel2_leq_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int128 income={}, bool predicate=1, bool polarity=0);
int64_uint64 v_i32_sel2_leq_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64_uint64 income, bool64 predicate, bool polarity=0);
int64_uint64 v_i32_sel2_leq_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64_uint64 income={}, bool predicate=1, bool polarity=0);

uint64_float64 v_u32_sel2_leq_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64_float64 income, bool64 predicate, bool polarity=0);
uint64_float64 v_u32_sel2_leq_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64_float64 income={}, bool predicate=1, bool polarity=0);
uint64_int64 v_u32_sel2_leq_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64_int64 income, bool64 predicate, bool polarity=0);
uint64_int64 v_u32_sel2_leq_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64_int64 income={}, bool predicate=1, bool polarity=0);
uint128 v_u32_sel2_leq_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint128 income, bool64 predicate, bool polarity=0);
uint128 v_u32_sel2_leq_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128_bfloat128 v_i16_sel2_leq_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128_bfloat128 income, bool128 predicate, bool polarity=0);
short128_bfloat128 v_i16_sel2_leq_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
short256 v_i16_sel2_leq_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short256 income, bool128 predicate, bool polarity=0);
short256 v_i16_sel2_leq_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short256 income={}, bool predicate=1, bool polarity=0);
short128_ushort128 v_i16_sel2_leq_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128_ushort128 income, bool128 predicate, bool polarity=0);
short128_ushort128 v_i16_sel2_leq_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128_ushort128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128_bfloat128 v_u16_sel2_leq_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128_bfloat128 income, bool128 predicate, bool polarity=0);
ushort128_bfloat128 v_u16_sel2_leq_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128_short128 v_u16_sel2_leq_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128_short128 income, bool128 predicate, bool polarity=0);
ushort128_short128 v_u16_sel2_leq_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128_short128 income={}, bool predicate=1, bool polarity=0);
ushort256 v_u16_sel2_leq_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort256 income, bool128 predicate, bool polarity=0);
ushort256 v_u16_sel2_leq_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort256 income={}, bool predicate=1, bool polarity=0);

char512 v_i8_sel2_leq_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char512 income, bool256 predicate, bool polarity=0);
char512 v_i8_sel2_leq_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char512 income={}, bool predicate=1, bool polarity=0);
char256_uchar256 v_i8_sel2_leq_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256_uchar256 income, bool256 predicate, bool polarity=0);
char256_uchar256 v_i8_sel2_leq_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256_uchar256 income={}, bool predicate=1, bool polarity=0);

uchar256_char256 v_u8_sel2_leq_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256_char256 income, bool256 predicate, bool polarity=0);
uchar256_char256 v_u8_sel2_leq_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256_char256 income={}, bool predicate=1, bool polarity=0);
uchar512 v_u8_sel2_leq_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar512 income, bool256 predicate, bool polarity=0);
uchar512 v_u8_sel2_leq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar512 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SEL2_GRT
//

/// @brief Represents SEL2_GRT instruction.
///
/// @param a           Source #1 to compare and select (SRC1).
/// @param b           Source #2 to compare and select (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Pair (c, a) or (d, b) with respect to the comparison of \p a and \p b.
///
/// @{
float128 v_f32_sel2_grt_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float128 income, bool64 predicate, bool polarity=0);
float128 v_f32_sel2_grt_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float128 income={}, bool predicate=1, bool polarity=0);
float64_int64 v_f32_sel2_grt_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64_int64 income, bool64 predicate, bool polarity=0);
float64_int64 v_f32_sel2_grt_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64_int64 income={}, bool predicate=1, bool polarity=0);
float64_uint64 v_f32_sel2_grt_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64_uint64 income, bool64 predicate, bool polarity=0);
float64_uint64 v_f32_sel2_grt_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64_uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat256 v_bf16_sel2_grt_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat256 income={}, bool predicate=1, bool polarity=0);
bfloat256 v_bf16_sel2_grt_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat256 income, bool128 predicate, bool polarity=0);
bfloat128_short128 v_bf16_sel2_grt_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_short128 income={}, bool predicate=1, bool polarity=0);
bfloat128_short128 v_bf16_sel2_grt_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_short128 income, bool128 predicate, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_grt_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_ushort128 income={}, bool predicate=1, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_grt_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_ushort128 income, bool128 predicate, bool polarity=0);
#endif


int64_float64 v_i32_sel2_grt_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64_float64 income, bool64 predicate, bool polarity=0);
int64_float64 v_i32_sel2_grt_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64_float64 income={}, bool predicate=1, bool polarity=0);
int128 v_i32_sel2_grt_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int128 income, bool64 predicate, bool polarity=0);
int128 v_i32_sel2_grt_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int128 income={}, bool predicate=1, bool polarity=0);
int64_uint64 v_i32_sel2_grt_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64_uint64 income, bool64 predicate, bool polarity=0);
int64_uint64 v_i32_sel2_grt_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64_uint64 income={}, bool predicate=1, bool polarity=0);

uint64_float64 v_u32_sel2_grt_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64_float64 income, bool64 predicate, bool polarity=0);
uint64_float64 v_u32_sel2_grt_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64_float64 income={}, bool predicate=1, bool polarity=0);
uint64_int64 v_u32_sel2_grt_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64_int64 income, bool64 predicate, bool polarity=0);
uint64_int64 v_u32_sel2_grt_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64_int64 income={}, bool predicate=1, bool polarity=0);
uint128 v_u32_sel2_grt_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint128 income, bool64 predicate, bool polarity=0);
uint128 v_u32_sel2_grt_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128_bfloat128 v_i16_sel2_grt_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128_bfloat128 income, bool128 predicate, bool polarity=0);
short128_bfloat128 v_i16_sel2_grt_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
short256 v_i16_sel2_grt_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short256 income, bool128 predicate, bool polarity=0);
short256 v_i16_sel2_grt_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short256 income={}, bool predicate=1, bool polarity=0);
short128_ushort128 v_i16_sel2_grt_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128_ushort128 income, bool128 predicate, bool polarity=0);
short128_ushort128 v_i16_sel2_grt_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128_ushort128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128_bfloat128 v_u16_sel2_grt_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128_bfloat128 income, bool128 predicate, bool polarity=0);
ushort128_bfloat128 v_u16_sel2_grt_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128_short128 v_u16_sel2_grt_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128_short128 income, bool128 predicate, bool polarity=0);
ushort128_short128 v_u16_sel2_grt_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128_short128 income={}, bool predicate=1, bool polarity=0);
ushort256 v_u16_sel2_grt_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort256 income, bool128 predicate, bool polarity=0);
ushort256 v_u16_sel2_grt_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort256 income={}, bool predicate=1, bool polarity=0);

char512 v_i8_sel2_grt_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char512 income, bool256 predicate, bool polarity=0);
char512 v_i8_sel2_grt_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char512 income={}, bool predicate=1, bool polarity=0);
char256_uchar256 v_i8_sel2_grt_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256_uchar256 income, bool256 predicate, bool polarity=0);
char256_uchar256 v_i8_sel2_grt_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256_uchar256 income={}, bool predicate=1, bool polarity=0);

uchar256_char256 v_u8_sel2_grt_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256_char256 income, bool256 predicate, bool polarity=0);
uchar256_char256 v_u8_sel2_grt_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256_char256 income={}, bool predicate=1, bool polarity=0);
uchar512 v_u8_sel2_grt_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar512 income, bool256 predicate, bool polarity=0);
uchar512 v_u8_sel2_grt_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar512 income={}, bool predicate=1, bool polarity=0);

/// @}

//
// ------ SEL2_GEQ
//

/// @brief Represents SEL2_GEQ instruction.
///
/// @param a           Source #1 to compare and select (SRC1).
/// @param b           Source #2 to compare and select (SRC2).
/// @param c           Source #1 to select (SRC3).
/// @param d           Source #2 to select (SRC4).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Pair (c, a) or (d, b) with respect to the comparison of \p a and \p b.
///
/// @{
float128 v_f32_sel2_geq_f32_vb(float64 a, float64 b, float64 c, float64 d, int switches, float128 income, bool64 predicate, bool polarity=0);
float128 v_f32_sel2_geq_f32_b(float64 a, float64 b, float64 c, float64 d, int switches=0, float128 income={}, bool predicate=1, bool polarity=0);
float64_int64 v_f32_sel2_geq_i32_vb(int64 a, int64 b, float64 c, float64 d, int switches, float64_int64 income, bool64 predicate, bool polarity=0);
float64_int64 v_f32_sel2_geq_i32_b(int64 a, int64 b, float64 c, float64 d, int switches=0, float64_int64 income={}, bool predicate=1, bool polarity=0);
float64_uint64 v_f32_sel2_geq_u32_vb(uint64 a, uint64 b, float64 c, float64 d, int switches, float64_uint64 income, bool64 predicate, bool polarity=0);
float64_uint64 v_f32_sel2_geq_u32_b(uint64 a, uint64 b, float64 c, float64 d, int switches=0, float64_uint64 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
bfloat256 v_bf16_sel2_geq_bf16_b(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat256 income={}, bool predicate=1, bool polarity=0);
bfloat256 v_bf16_sel2_geq_bf16_vb(bfloat128 a, bfloat128 b, bfloat128 c, bfloat128 d, int switches, bfloat256 income, bool128 predicate, bool polarity=0);
bfloat128_short128 v_bf16_sel2_geq_i16_b(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_short128 income={}, bool predicate=1, bool polarity=0);
bfloat128_short128 v_bf16_sel2_geq_i16_vb(short128 a, short128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_short128 income, bool128 predicate, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_geq_u16_b(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches=0, bfloat128_ushort128 income={}, bool predicate=1, bool polarity=0);
bfloat128_ushort128 v_bf16_sel2_geq_u16_vb(ushort128 a, ushort128 b, bfloat128 c, bfloat128 d, int switches, bfloat128_ushort128 income, bool128 predicate, bool polarity=0);
#endif


int64_float64 v_i32_sel2_geq_f32_vb(float64 a, float64 b, int64 c, int64 d, int switches, int64_float64 income, bool64 predicate, bool polarity=0);
int64_float64 v_i32_sel2_geq_f32_b(float64 a, float64 b, int64 c, int64 d, int switches=0, int64_float64 income={}, bool predicate=1, bool polarity=0);
int128 v_i32_sel2_geq_i32_vb(int64 a, int64 b, int64 c, int64 d, int switches, int128 income, bool64 predicate, bool polarity=0);
int128 v_i32_sel2_geq_i32_b(int64 a, int64 b, int64 c, int64 d, int switches=0, int128 income={}, bool predicate=1, bool polarity=0);
int64_uint64 v_i32_sel2_geq_u32_vb(uint64 a, uint64 b, int64 c, int64 d, int switches, int64_uint64 income, bool64 predicate, bool polarity=0);
int64_uint64 v_i32_sel2_geq_u32_b(uint64 a, uint64 b, int64 c, int64 d, int switches=0, int64_uint64 income={}, bool predicate=1, bool polarity=0);

uint64_float64 v_u32_sel2_geq_f32_vb(float64 a, float64 b, uint64 c, uint64 d, int switches, uint64_float64 income, bool64 predicate, bool polarity=0);
uint64_float64 v_u32_sel2_geq_f32_b(float64 a, float64 b, uint64 c, uint64 d, int switches=0, uint64_float64 income={}, bool predicate=1, bool polarity=0);
uint64_int64 v_u32_sel2_geq_i32_vb(int64 a, int64 b, uint64 c, uint64 d, int switches, uint64_int64 income, bool64 predicate, bool polarity=0);
uint64_int64 v_u32_sel2_geq_i32_b(int64 a, int64 b, uint64 c, uint64 d, int switches=0, uint64_int64 income={}, bool predicate=1, bool polarity=0);
uint128 v_u32_sel2_geq_u32_vb(uint64 a, uint64 b, uint64 c, uint64 d, int switches, uint128 income, bool64 predicate, bool polarity=0);
uint128 v_u32_sel2_geq_u32_b(uint64 a, uint64 b, uint64 c, uint64 d, int switches=0, uint128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
short128_bfloat128 v_i16_sel2_geq_bf16_vb(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches, short128_bfloat128 income, bool128 predicate, bool polarity=0);
short128_bfloat128 v_i16_sel2_geq_bf16_b(bfloat128 a, bfloat128 b, short128 c, short128 d, int switches=0, short128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
short256 v_i16_sel2_geq_i16_vb(short128 a, short128 b, short128 c, short128 d, int switches, short256 income, bool128 predicate, bool polarity=0);
short256 v_i16_sel2_geq_i16_b(short128 a, short128 b, short128 c, short128 d, int switches=0, short256 income={}, bool predicate=1, bool polarity=0);
short128_ushort128 v_i16_sel2_geq_u16_vb(ushort128 a, ushort128 b, short128 c, short128 d, int switches, short128_ushort128 income, bool128 predicate, bool polarity=0);
short128_ushort128 v_i16_sel2_geq_u16_b(ushort128 a, ushort128 b, short128 c, short128 d, int switches=0, short128_ushort128 income={}, bool predicate=1, bool polarity=0);

#if defined(__gaudi_plus__)
ushort128_bfloat128 v_u16_sel2_geq_bf16_vb(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches, ushort128_bfloat128 income, bool128 predicate, bool polarity=0);
ushort128_bfloat128 v_u16_sel2_geq_bf16_b(bfloat128 a, bfloat128 b, ushort128 c, ushort128 d, int switches=0, ushort128_bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
ushort128_short128 v_u16_sel2_geq_i16_vb(short128 a, short128 b, ushort128 c, ushort128 d, int switches, ushort128_short128 income, bool128 predicate, bool polarity=0);
ushort128_short128 v_u16_sel2_geq_i16_b(short128 a, short128 b, ushort128 c, ushort128 d, int switches=0, ushort128_short128 income={}, bool predicate=1, bool polarity=0);
ushort256 v_u16_sel2_geq_u16_vb(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches, ushort256 income, bool128 predicate, bool polarity=0);
ushort256 v_u16_sel2_geq_u16_b(ushort128 a, ushort128 b, ushort128 c, ushort128 d, int switches=0, ushort256 income={}, bool predicate=1, bool polarity=0);

char512 v_i8_sel2_geq_i8_vb(char256 a, char256 b, char256 c, char256 d, int switches, char512 income, bool256 predicate, bool polarity=0);
char512 v_i8_sel2_geq_i8_b(char256 a, char256 b, char256 c, char256 d, int switches=0, char512 income={}, bool predicate=1, bool polarity=0);
char256_uchar256 v_i8_sel2_geq_u8_vb(uchar256 a, uchar256 b, char256 c, char256 d, int switches, char256_uchar256 income, bool256 predicate, bool polarity=0);
char256_uchar256 v_i8_sel2_geq_u8_b(uchar256 a, uchar256 b, char256 c, char256 d, int switches=0, char256_uchar256 income={}, bool predicate=1, bool polarity=0);

uchar256_char256 v_u8_sel2_geq_i8_vb(char256 a, char256 b, uchar256 c, uchar256 d, int switches, uchar256_char256 income, bool256 predicate, bool polarity=0);
uchar256_char256 v_u8_sel2_geq_i8_b(char256 a, char256 b, uchar256 c, uchar256 d, int switches=0, uchar256_char256 income={}, bool predicate=1, bool polarity=0);
uchar512 v_u8_sel2_geq_u8_vb(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches, uchar512 income, bool256 predicate, bool polarity=0);
uchar512 v_u8_sel2_geq_u8_b(uchar256 a, uchar256 b, uchar256 c, uchar256 d, int switches=0, uchar512 income={}, bool predicate=1, bool polarity=0);

/// @}


//
// ------ SET_INDX
//


/// @brief Represents SET_INDX instruction.
///
/// @param value       Value assigned to selected dimensions (SRC1).
/// @param ndx         Source tensor coordinates.
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of the instruction.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Tensor coordinates after assignment.
///
/// @{
int5 set_indx(int value, int5 ndx, const int dimmask, int switches, bool predicate=1, bool polarity=0);
/// @}


//
// ------ SHL
//


/// @brief Represents SHL instruction.
///
/// @param a           The value to shift (SRC1).
/// @param b           The number of bits to shift left (SRC2).
/// @param switches    Switches of SHL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_shl(float a, int8_t b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_shl(bf16 a, uint8_t b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_shl(int32_t a, int8_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_shl(uint32_t a, int8_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_shl(int16_t a, int8_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_shl(uint16_t a, int8_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_shl(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_shl(uint8_t a, int8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);

float64 v_f32_shl_b(float64 a, int64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_shl_vb(float64 a, int64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_shl_b(bfloat128 a, short128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_shl_vb(bfloat128 a, short128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
int64 v_i32_shl_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_shl_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_shl_b(uint64 a, int64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_shl_vb(uint64 a, int64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_shl_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_shl_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_shl_b(ushort128 a, short128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_shl_vb(ushort128 a, short128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_shl_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_shl_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_shl_b(uchar256 a, char256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_shl_vb(uchar256 a, char256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents SHL instruction for int5 operands.
///
/// @param a           The value to shift (SRC1).
/// @param b           The number of bits to shift left (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of SHL instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
#endif
/// @{
#if defined(__gaudi_plus__)
int5 i_i32_shl(int5 a, int5 b, int dimmask, const int switches, int5 income, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ SHR
//


/// @brief Represents SHR instruction.
///
/// @param a           The value to shift (SRC1).
/// @param b           The number of bits to shift right (SRC2).
/// @param switches    Switches of SHR instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_shr(float a, int8_t b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_shr(bf16 a, uint8_t b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_shr(int32_t a, int8_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_shr(uint32_t a, int8_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_shr(int16_t a, int8_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_shr(uint16_t a, int8_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_shr(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_shr(uint8_t a, int8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);

float64 v_f32_shr_b(float64 a, int64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
float64 v_f32_shr_vb(float64 a, int64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_shr_b(bfloat128 a, short128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_shr_vb(bfloat128 a, short128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
int64 v_i32_shr_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
int64 v_i32_shr_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_shr_b(uint64 a, int64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_shr_vb(uint64 a, int64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_shr_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_shr_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_shr_b(ushort128 a, short128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_shr_vb(ushort128 a, short128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_shr_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_shr_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_shr_b(uchar256 a, char256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_shr_vb(uchar256 a, char256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents SHR instruction for int5 operands.
///
/// @param a           The value to shift (SRC1).
/// @param b           The number of bits to shift right (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of SHR instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
#endif
/// @{
#if defined(__gaudi_plus__)
int5 i_i32_shr(int5 a, int5 b, int dimmask, const int switches, int5 income, bool predicate=1, bool polarity=0);
#endif
/// @}


//
// ------ SHUFFLE
//

/// @brief Represents SHUFFLE instruction.
///
/// @param a           Value to shuffle (SRC1).
/// @param b           SHUFFLE directions (SRC2).
/// @param switches    Instruction switches.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Reshuffled vector.
///
/// For each element in DEST b encodes the src element in the following format \n
/// | bits        | 0-4            | 5            | 6        | 7               |
/// |-------------|----------------|--------------|----------|-----------------|
/// | description | source-element | source-group | reserved | shuffle-enabled |
///
/// Only if shuffle-enabled bit is set the output element should be written.
///
/// \c DEST[i]=a[b[i]]
///
/// @{
float64 v_f32_shuffle_b(float64 a, uchar256 b, int switches, float64 income, bool predicate=1, bool polarity=0);
float64 v_f32_shuffle_vb(float64 a, uchar256 b, int switches, float64 income, bool64 predicate, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_shuffle_b(bfloat128 a, uchar256 b, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_shuffle_vb(bfloat128 a, uchar256 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
int64 v_i32_shuffle_b(int64 a, uchar256 b, int switches, int64 income, bool predicate=1, bool polarity=0);
int64 v_i32_shuffle_vb(int64 a, uchar256 b, int switches, int64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_shuffle_b(uint64 a, uchar256 b, int switches, uint64 income, bool predicate=1, bool polarity=0);
uint64 v_u32_shuffle_vb(uint64 a, uchar256 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
short128 v_i16_shuffle_b(short128 a, uchar256 b, int switches, short128 income, bool predicate=1, bool polarity=0);
short128 v_i16_shuffle_vb(short128 a, uchar256 b, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_shuffle_b(ushort128 a, uchar256 b, int switches, ushort128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_shuffle_vb(ushort128 a, uchar256 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_shuffle_b(char256 a, uchar256 b, int switches, char256 income, bool predicate=1, bool polarity=0);
char256 v_i8_shuffle_vb(char256 a, uchar256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_shuffle_b(uchar256 a, uchar256 b, int switches, uchar256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_shuffle_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}

//
// ------ ST_L
//

/// @brief Represents ST_L instruction.
///
/// @param addr        Address to write to (SRC1).
/// @param value       Value to write (SRC2).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
/// \par Allowed switches are:
///     \li SW_MMIO - When set store to MMIO, else - store to SLM.
///
/// @{
void s_f32_st_l(uint32_t addr, float value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void s_bf16_st_l(uint32_t addr, bf16 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void s_i32_st_l(uint32_t addr, int32_t value, int switches=0, bool predicate=1, bool polarity=0);
void s_u32_st_l(uint32_t addr, uint32_t value, int switches=0, bool predicate=1, bool polarity=0);
void s_i16_st_l(uint32_t addr, int16_t value, int switches=0, bool predicate=1, bool polarity=0);
void s_u16_st_l(uint32_t addr, uint16_t value, int switches=0, bool predicate=1, bool polarity=0);
void s_i8_st_l(uint32_t addr, int8_t value, int switches=0, bool predicate=1, bool polarity=0);
void s_u8_st_l(uint32_t addr, uint8_t value, int switches=0, bool predicate=1, bool polarity=0);
void s_i1_st_l(uint32_t addr, bool value, int switches=0, bool predicate=1, bool polarity=0);
/// @}


//
// ------ ST_L_V
//

/// @brief Represents ST_L_V instruction.
///
/// @param addr        Address to write to (SRC1).
/// @param value       Value to write (SRC2).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
///
/// @{
void v_f32_st_l_v(uint32_t addr, float64 value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void v_bf16_st_l_v(uint32_t addr, bfloat128 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void v_i32_st_l_v(uint32_t addr, int64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_l_v(uint32_t addr, uint64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_l_v(uint32_t addr, short128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_l_v(uint32_t addr, ushort128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_l_v(uint32_t addr, char256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_l_v(uint32_t addr, uchar256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_l_v(uint32_t addr, bool256 value, int switches=0, bool predicate=1, bool polarity=0);
/// @}


//
// ------ ST_L_V_LOW
//

/// @brief Represents ST_L_V_LOW instruction.
///
/// @param addr        Address to write to (SRC1).
/// @param value       Value to write (SRC2).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
///
/// @{
void v_f32_st_l_v_low(uint32_t addr, float64 value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void v_bf16_st_l_v_low(uint32_t addr, bfloat128 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void v_i32_st_l_v_low(uint32_t addr, int64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_l_v_low(uint32_t addr, uint64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_l_v_low(uint32_t addr, short128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_l_v_low(uint32_t addr, ushort128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_l_v_low(uint32_t addr, char256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_l_v_low(uint32_t addr, uchar256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_l_v_low(uint32_t addr, bool256 value, int switches=0, bool predicate=1, bool polarity=0);
/// @}


//
// ------ ST_L_V_HIGH
//

/// @brief Represents ST_L_V_HIGH instruction.
///
/// @param addr        Address to write to (SRC1).
/// @param value       Value to write (SRC2).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
///
/// @{
void v_f32_st_l_v_high(uint32_t addr, float64 value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void v_bf16_st_l_v_high(uint32_t addr, bfloat128 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void v_i32_st_l_v_high(uint32_t addr, int64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_l_v_high(uint32_t addr, uint64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_l_v_high(uint32_t addr, short128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_l_v_high(uint32_t addr, ushort128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_l_v_high(uint32_t addr, char256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_l_v_high(uint32_t addr, uchar256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_l_v_high(uint32_t addr, bool256 value, int switches=0, bool predicate=1, bool polarity=0);
/// @}

//
// ------ ST_TNSR
//

/// @brief Represents ST_TNSR instruction.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param value       Value to write (SRC3).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
#if defined(__gaudi_plus__)
/// \par Allowed switches are:
///     \li SW_PACK - When set, store pipe will perform packing according to PACK_DT.
#endif
///
/// @{
void v_f32_st_tnsr(int5 ndx, const int8_t tensor, float64 value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void v_bf16_st_tnsr(int5 ndx, const int8_t tensor, bfloat128 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void v_i32_st_tnsr(int5 ndx, const int8_t tensor, int64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_tnsr(int5 ndx, const int8_t tensor, uint64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_tnsr(int5 ndx, const int8_t tensor, short128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_tnsr(int5 ndx, const int8_t tensor, ushort128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_tnsr(int5 ndx, const int8_t tensor, char256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_tnsr(int5 ndx, const int8_t tensor, uchar256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_tnsr(int5 ndx, const int8_t tensor, bool256 value, int switches=0, bool predicate=1, bool polarity=0);
/// @}


#if defined(__gaudi_plus__)
/// @brief Represents ST_TNSR instruction with PARTIAL switch.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param value       Value to write (SRC3).
/// @param size        Size in elements minus 1.
/// @param offset      Offset in elements.
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
/// \par Allowed switches are:
///     \li SW_PACK - When set, store pipe will perform packing according to PACK_DT.
#endif
///
/// @{
#if defined(__gaudi_plus__)
void v_f32_st_tnsr_partial(int5 ndx, int8_t tensor, float64 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_bf16_st_tnsr_partial(int5 ndx, int8_t tensor, bfloat128 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_i32_st_tnsr_partial(int5 ndx, int8_t tensor, int64 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_tnsr_partial(int5 ndx, int8_t tensor, uint64 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_tnsr_partial(int5 ndx, int8_t tensor, short128 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_tnsr_partial(int5 ndx, int8_t tensor, ushort128 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_tnsr_partial(int5 ndx, int8_t tensor, char256 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_tnsr_partial(int5 ndx, int8_t tensor, uchar256 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_tnsr_partial(int5 ndx, int8_t tensor, bool256 value, int8_t size, int8_t offset, int switches=0, bool predicate=1, bool polarity=0);
#endif
/// @}

//
// ------ ST_TNSR_HIGH
//

/// @brief Represents ST_TNSR_HIGH instruction.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param value       Value to write (SRC3).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
#if defined(__gaudi_plus__)
/// \par Allowed switches are:
///     \li SW_PACK - When set, store pipe will perform packing according to PACK_DT.
#endif
///
/// @{
void v_f32_st_tnsr_high(int5 ndx, const int8_t tensor, float64 value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void v_bf16_st_tnsr_high(int5 ndx, const int8_t tensor, bfloat128 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void v_i32_st_tnsr_high(int5 ndx, const int8_t tensor, int64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_tnsr_high(int5 ndx, const int8_t tensor, uint64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_tnsr_high(int5 ndx, const int8_t tensor, short128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_tnsr_high(int5 ndx, const int8_t tensor, ushort128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_tnsr_high(int5 ndx, const int8_t tensor, char256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_tnsr_high(int5 ndx, const int8_t tensor, uchar256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_tnsr_high(int5 ndx, const int8_t tensor, bool256 value, int switches=0, bool predicate=1, bool polarity=0);
/// @}

//
// ------ ST_TNSR_LOW
//

/// @brief Represents ST_TNSR_LOW instruction.
///
/// @param ndx         Tensor coordinates (SRC1).
/// @param tensor      Tensor number.
/// @param value       Value to write (SRC3).
/// @param switches    Instruction switches.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
///
#if defined(__gaudi_plus__)
/// \par Allowed switches are:
///     \li SW_PACK - When set, store pipe will perform packing according to PACK_DT.
#endif
///
/// @{
void v_f32_st_tnsr_low(int5 ndx, const int8_t tensor, float64 value, int switches=0, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
void v_bf16_st_tnsr_low(int5 ndx, const int8_t tensor, bfloat128 value, int switches=0, bool predicate=1, bool polarity=0);
#endif
void v_i32_st_tnsr_low(int5 ndx, const int8_t tensor, int64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u32_st_tnsr_low(int5 ndx, const int8_t tensor, uint64 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i16_st_tnsr_low(int5 ndx, const int8_t tensor, short128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u16_st_tnsr_low(int5 ndx, const int8_t tensor, ushort128 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i8_st_tnsr_low(int5 ndx, const int8_t tensor, char256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_u8_st_tnsr_low(int5 ndx, const int8_t tensor, uchar256 value, int switches=0, bool predicate=1, bool polarity=0);
void v_i1_st_tnsr_low(int5 ndx, const int8_t tensor, bool256 value, int switches=0, bool predicate=1, bool polarity=0);
/// @}

//
// ------ SUB
//

/// @brief Represents SUB instruction.
///
/// @param a           The first SRC operand to SUB (SRC1).
/// @param b           The second SRC operand to SUB (SRC2).
/// @param switches    Switches of SUB instructions.
/// @param income      Income value of DEST.
/// @param switches    Switches of SUB instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of the operation.
///
/// \par Allowed switches are:
///     \li SW_SAT - Saturate (integer types only).
///     \li SW_NEG - Negates the destination after the operation.
///
/// @{
float s_f32_sub(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_sub(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_sub(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_sub(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_sub(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_sub(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_sub(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_sub(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);

float64 v_f32_sub_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_sub_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_sub_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_sub_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_sub_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_sub_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_sub_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_sub_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_sub_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_sub_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_sub_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_sub_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_sub_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_sub_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_sub_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_sub_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
/// @}


/// @brief Represents SUB instruction for int5.
///
/// @param a           The first SRC operand to SUB (SRC1).
/// @param b           The second SRC operand to SUB (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of SUB instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of the operation.
///
/// @{
int5 i_i32_sub(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}

//
// ------ UNPACK
//

/// @brief Represents UNPACK instruction.
///
/// @param a           Value to be transformed (SRC1).
/// @param switches    Switches of the instruction.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Unpacked value.
///
/// \par Allowed switches are:
///     - [GROUP_SOURCE]
///         \li SW_GROUP_0
///         \li SW_GROUP_1
///     - [ELEMENT_STRIDE]
///         \li SW_STRIDE_2 - Every second element is valid.
///         \li SW_STRIDE_4 - Every forth element is valid.
///     - [GROUP_HALF]
///         \li SW_GROUP_HALF_0 - The lower half of the input group is taken.
///         \li SW_GROUP_HALF_1 - The upper half of the input group is taken.
///
/// @{
#if defined(__gaudi_plus__)
bfloat128 v_bf16_unpack_b(bfloat128 a, int switches, bfloat128 income, bool predicate=1, bool polarity=0);
bfloat128 v_bf16_unpack_vb(bfloat128 a, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
#endif
short128 v_i16_unpack_b(short128 a, int switches, short128 income, bool predicate=1, bool polarity=0);
short128 v_i16_unpack_vb(short128 a, int switches, short128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_unpack_b(ushort128 a, int switches, ushort128 income, bool predicate=1, bool polarity=0);
ushort128 v_u16_unpack_vb(ushort128 a, int switches, ushort128 income, bool128 predicate, bool polarity=0);
char256 v_i8_unpack_b(char256 a, int switches, char256 income, bool predicate=1, bool polarity=0);
char256 v_i8_unpack_vb(char256 a, int switches, char256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_unpack_b(uchar256 a, int switches, uchar256 income, bool predicate=1, bool polarity=0);
uchar256 v_u8_unpack_vb(uchar256 a, int switches, uchar256 income, bool256 predicate, bool polarity=0);
/// @}


//
// ------ XOR
//

/// @brief Represents XOR instruction.
///
/// @param a           The first SRC operand to XOR (SRC1).
/// @param b           The second SRC operand to XOR (SRC2).
/// @param switches    Switches of XOR instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value for the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
float s_f32_xor(float a, float b, int switches=0, float income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bf16 s_bf16_xor(bf16 a, bf16 b, int switches=0, bf16 income={}, bool predicate=1, bool polarity=0);
#endif
int32_t s_i32_xor(int32_t a, int32_t b, int switches=0, int32_t income={}, bool predicate=1, bool polarity=0);
uint32_t s_u32_xor(uint32_t a, uint32_t b, int switches=0, uint32_t income={}, bool predicate=1, bool polarity=0);
int16_t s_i16_xor(int16_t a, int16_t b, int switches=0, int16_t income={}, bool predicate=1, bool polarity=0);
uint16_t s_u16_xor(uint16_t a, uint16_t b, int switches=0, uint16_t income={}, bool predicate=1, bool polarity=0);
int8_t s_i8_xor(int8_t a, int8_t b, int switches=0, int8_t income={}, bool predicate=1, bool polarity=0);
uint8_t s_u8_xor(uint8_t a, uint8_t b, int switches=0, uint8_t income={}, bool predicate=1, bool polarity=0);
bool s_i1_xor(bool a, bool b, int switches=0, bool income={}, bool predicate=1, bool polarity=0);

float64 v_f32_xor_vb(float64 a, float64 b, int switches, float64 income, bool64 predicate, bool polarity=0);
float64 v_f32_xor_b(float64 a, float64 b, int switches=0, float64 income={}, bool predicate=1, bool polarity=0);
#if defined(__gaudi_plus__)
bfloat128 v_bf16_xor_vb(bfloat128 a, bfloat128 b, int switches, bfloat128 income, bool128 predicate, bool polarity=0);
bfloat128 v_bf16_xor_b(bfloat128 a, bfloat128 b, int switches=0, bfloat128 income={}, bool predicate=1, bool polarity=0);
#endif
int64 v_i32_xor_vb(int64 a, int64 b, int switches, int64 income, bool64 predicate, bool polarity=0);
int64 v_i32_xor_b(int64 a, int64 b, int switches=0, int64 income={}, bool predicate=1, bool polarity=0);
uint64 v_u32_xor_vb(uint64 a, uint64 b, int switches, uint64 income, bool64 predicate, bool polarity=0);
uint64 v_u32_xor_b(uint64 a, uint64 b, int switches=0, uint64 income={}, bool predicate=1, bool polarity=0);
short128 v_i16_xor_vb(short128 a, short128 b, int switches, short128 income, bool128 predicate, bool polarity=0);
short128 v_i16_xor_b(short128 a, short128 b, int switches=0, short128 income={}, bool predicate=1, bool polarity=0);
ushort128 v_u16_xor_vb(ushort128 a, ushort128 b, int switches, ushort128 income, bool128 predicate, bool polarity=0);
ushort128 v_u16_xor_b(ushort128 a, ushort128 b, int switches=0, ushort128 income={}, bool predicate=1, bool polarity=0);
char256 v_i8_xor_vb(char256 a, char256 b, int switches, char256 income, bool256 predicate, bool polarity=0);
char256 v_i8_xor_b(char256 a, char256 b, int switches=0, char256 income={}, bool predicate=1, bool polarity=0);
uchar256 v_u8_xor_vb(uchar256 a, uchar256 b, int switches, uchar256 income, bool256 predicate, bool polarity=0);
uchar256 v_u8_xor_b(uchar256 a, uchar256 b, int switches=0, uchar256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_xor_b(bool256 a, bool256 b, int switches=0, bool256 income={}, bool predicate=1, bool polarity=0);
bool256 v_i1_xor_vb(bool256 a, bool256 b, int switches, bool256 income, bool256 predicate, bool polarity=0);
/// @}


/// @brief Represents scalar XOR instruction for int5.
///
/// @param a           The first SRC operand to XOR (SRC1).
/// @param b           The second SRC operand to XOR (SRC2).
/// @param dimmask     Selects IRF lanes participated in the operation.
/// @param switches    Switches of XOR instructions.
/// @param income      Income value of DEST.
/// @param predicate   Predicate value fxor the instruction.
/// @param polarity    True if polarity of the predicate is inverted.
/// @return Result of operation.
///
/// @{
int5 i_i32_xor(int5 a, int5 b, const int dimmask, int switches, int5 income, bool predicate=1, bool polarity=0);
/// @}
