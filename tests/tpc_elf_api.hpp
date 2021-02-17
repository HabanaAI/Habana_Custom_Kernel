/*****************************************************************************
 * Copyright (C) 2019 HabanaLabs, Ltd.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Tzachi Cohen <tcohen@habana.ai>
 ******************************************************************************
 */
#ifndef TPC_ELF_API_HPP
#define TPC_ELF_API_HPP

#include <gc_interface.h>

extern "C"{
namespace TpcElfTools
{

enum TpcElfStatus
{
    TPC_ELF_SUCCESS                            = 0,
    TPC_ELF_INVALID_ELF_BUFFER                 = 1,
    TPC_ELF_SECTION_NOT_FOUND                  = 2,
    TPC_ELF_UNSUPPORTED                        = 3,
} ;

struct TPCProgramHeader
{
    uint32_t    version;                // version of header
    bool        specialFunctionUsed;
    bool        printfUsed;
    bool        lockUnlock;
    bool        mmioUse;
    uint16_t    march;
    bool        reserved_temp[16];
    uint16_t    scalarLoad;
    uint16_t    rmwStore;
    uint32_t    reserved[58];
} ;
/*!
 ***************************************************************************************************
 *   @brief Returns pointer and size of TPC binary from elf buffer
 *
 *   @param pElf            [in]    pointer to elf buffer
 *   @param size            [in]    size of elf buffer
 *   @param pTpcBinary      [out]   Returned pointer to TPC binary on elf buffer
 *   @param tpcBinarySize   [out]   Returned size to TPC binary
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus ExtractTpcBinaryFromElf(   const void*  pElf,
                                        uint32_t     elfSize,
                                        void*&       pTpcBinary,
                                        uint32_t&    tpcBinarySize);



/*!
 ***************************************************************************************************
 *   @brief Returns TPC program header from elf buffer
 *
 *   @param pElf            [in]    Pointer to elf buffer
 *   @param size            [in]    Size of elf buffer
 *   @param programHeader   [out]   program header structure.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus ExtractTpcProgramHeaderFromElf(    const void*     pElf,
                                                uint32_t        elfSize,
                                                TPCProgramHeader&  programHeader);


/*!
 ***************************************************************************************************
 *   @brief Returns estimated cycle count of program execution.
 *
 *   @param pHabanaKernelParam [in]    pointer to HabanaKernelParam
 *   @param HabanaKernelInstantiation_t [in]    pointer to HabanaKernelInstantiation
 *   @param cycleCount         [out]   Estimated execution time in cycles.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus GetTpcProgramCycleCount(gcapi::HabanaKernelParams_t *pHabanaKernelParam,
                                     gcapi::HabanaKernelInstantiation_t *pHabanaKernelInstantiation,
                                     uint32_t &cycleCount);


/*!
 ***************************************************************************************************
 *   @brief Fills index space mapping in kernel instatiation struct based on SCEV analysis
 *
 *   @param params        [in]    kernel instantiation struct which holds the elf buffer.
 *   @param instance      [out]   program instance struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
TpcElfStatus GetTpcProgramIndexSpaceMapping(   gcapi::HabanaKernelParams_t *          params,
                                               gcapi::HabanaKernelInstantiation_t*    instance);




} // end of TpcElfTools
}
#endif /* TPC_ELF_API_HPP */
