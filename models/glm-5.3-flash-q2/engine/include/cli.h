#ifndef DS4_HELP_H
#define DS4_HELP_H

#include <stdio.h>

typedef enum {
    DS4_HELP_DS4,
    DS4_HELP_SERVER,
    DS4_HELP_AGENT,
    DS4_HELP_BENCH,
    DS4_HELP_EVAL,
} cli_tool;

void cli_print(FILE *fp, cli_tool tool, const char *topic);

#endif
