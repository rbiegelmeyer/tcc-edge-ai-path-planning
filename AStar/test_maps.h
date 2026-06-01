/**
 * @file test_maps.h
 * @brief Mazes for A* validation on STM32H743.
 *
 * Generated from W064xH064_D03_S100000_E100100.csv
 *
 * Cell values: 0=empty 1=wall 2=path 3=start 4=end
 */

#ifndef TEST_MAPS_H
#define TEST_MAPS_H

#include <stdint.h>

#define TEST_MAP_COUNT  100
#define TEST_MAP_HEIGHT 64
#define TEST_MAP_WIDTH  64

/** Metadata associated with each test maze. */
typedef struct {
    uint32_t id;
    int16_t  start_row;  /**< Row of the start cell (y). */
    int16_t  start_col;  /**< Column of the start cell (x). */
    int16_t  end_row;    /**< Row of the goal cell (y). */
    int16_t  end_col;    /**< Column of the goal cell (x). */
} TestMapMeta;

extern const int8_t map_to_solve_0[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_1[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_2[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_3[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_4[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_5[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_6[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_7[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_8[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_9[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_10[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_11[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_12[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_13[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_14[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_15[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_16[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_17[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_18[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_19[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_20[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_21[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_22[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_23[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_24[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_25[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_26[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_27[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_28[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_29[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_30[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_31[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_32[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_33[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_34[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_35[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_36[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_37[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_38[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_39[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_40[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_41[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_42[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_43[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_44[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_45[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_46[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_47[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_48[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_49[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_50[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_51[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_52[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_53[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_54[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_55[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_56[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_57[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_58[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_59[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_60[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_61[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_62[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_63[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_64[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_65[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_66[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_67[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_68[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_69[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_70[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_71[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_72[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_73[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_74[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_75[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_76[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_77[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_78[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_79[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_80[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_81[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_82[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_83[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_84[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_85[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_86[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_87[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_88[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_89[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_90[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_91[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_92[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_93[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_94[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_95[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_96[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_97[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_98[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];
extern const int8_t map_to_solve_99[TEST_MAP_HEIGHT][TEST_MAP_WIDTH];

extern const TestMapMeta test_maps_meta[TEST_MAP_COUNT];

#endif /* TEST_MAPS_H */
