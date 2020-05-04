import pdb
from enum import Enum

ComAction = Enum('Action', (
    'ACTION_COM_SEL_0_COLOR_0_CONVERT_COLOR',   # 1
    'ACTION_COM_SEL_0_COLOR_1_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_2_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_3_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_4_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_5_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_6_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_7_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_8_CONVERT_COLOR',
    'ACTION_COM_SEL_0_COLOR_9_CONVERT_COLOR',

    'ACTION_COM_SEL_1_COLOR_0_CONVERT_COLOR',   # 11
    'ACTION_COM_SEL_1_COLOR_1_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_2_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_3_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_4_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_5_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_6_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_7_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_8_CONVERT_COLOR',
    'ACTION_COM_SEL_1_COLOR_9_CONVERT_COLOR',

    'ACTION_COM_SEL_2_COLOR_0_CONVERT_COLOR',   # 21
    'ACTION_COM_SEL_2_COLOR_1_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_2_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_3_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_4_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_5_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_6_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_7_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_8_CONVERT_COLOR',
    'ACTION_COM_SEL_2_COLOR_9_CONVERT_COLOR',

    'ACTION_COM_SEL_3_COLOR_0_CONVERT_COLOR',   # 31
    'ACTION_COM_SEL_3_COLOR_1_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_2_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_3_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_4_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_5_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_6_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_7_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_8_CONVERT_COLOR',
    'ACTION_COM_SEL_3_COLOR_9_CONVERT_COLOR',

    'ACTION_COM_SEL_4_COLOR_0_CONVERT_COLOR',   # 41
    'ACTION_COM_SEL_4_COLOR_1_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_2_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_3_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_4_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_5_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_6_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_7_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_8_CONVERT_COLOR',
    'ACTION_COM_SEL_4_COLOR_9_CONVERT_COLOR',

    'ACTION_COM_SEL_0_TOP_MOVE_UNTIL_COLISSION',    #51
    'ACTION_COM_SEL_0_BOTTOM_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_0_LEFT_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_0_RIGHT_MOVE_UNTIL_COLISSION',

    'ACTION_COM_SEL_1_TOP_MOVE_UNTIL_COLISSION',    # 55
    'ACTION_COM_SEL_1_BOTTOM_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_1_LEFT_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_1_RIGHT_MOVE_UNTIL_COLISSION',

    'ACTION_COM_SEL_2_TOP_MOVE_UNTIL_COLISSION',    # 59
    'ACTION_COM_SEL_2_BOTTOM_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_2_LEFT_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_2_RIGHT_MOVE_UNTIL_COLISSION',

    'ACTION_COM_SEL_3_TOP_MOVE_UNTIL_COLISSION',    # 63
    'ACTION_COM_SEL_3_BOTTOM_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_3_LEFT_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_3_RIGHT_MOVE_UNTIL_COLISSION',

    'ACTION_COM_SEL_4_TOP_MOVE_UNTIL_COLISSION',    # 67
    'ACTION_COM_SEL_4_BOTTOM_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_4_LEFT_MOVE_UNTIL_COLISSION',
    'ACTION_COM_SEL_4_RIGHT_MOVE_UNTIL_COLISSION',

    'ACTION_COM_SEL_0_POINT_HORI_SPREAD',             # 71
    'ACTION_COM_SEL_0_POINT_VERT_SPREAD',

    'ACTION_COM_SEL_1_POINT_HORI_SPEAD',
    'ACTION_COM_SEL_1_POINT_VERT_SPEAD',

    'ACTION_COM_SEL_2_POINT_HORI_SPEAD',
    'ACTION_COM_SEL_2_POINT_VERT_SPEAD',

    'ACTION_COM_SEL_3_POINT_HORI_SPEAD',
    'ACTION_COM_SEL_3_POINT_VERT_SPEAD',

    'ACTION_COM_SEL_4_POINT_HORI_SPEAD',
    'ACTION_COM_SEL_4_POINT_VERT_SPEAD',

    'ACTION_COM_NUM'
))