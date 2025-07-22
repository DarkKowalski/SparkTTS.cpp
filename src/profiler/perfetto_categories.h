#ifdef ENABLE_PERFETTO
#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("synthesizer")
        .SetDescription("Synthesizer workload"),
    perfetto::Category("transformer")
        .SetDescription("Transformer workload"),
    perfetto::Category("audio_tokenizer")
        .SetDescription("AudioTokenizer workload"),
    perfetto::Category("audio_detokenizer")
        .SetDescription("AudioDetokenizer workload"),
    perfetto::Category("misc")
        .SetDescription("Misc workload"));

#endif // ENABLE_PERFETTO
