import asyncio

import pytest

from utils.rate_limiter import PROVIDER_DEFAULTS, get_semaphore, reset_semaphore


class TestProviderAwareDefaults:
    def setup_method(self):
        reset_semaphore()

    def teardown_method(self):
        reset_semaphore()

    def test_anthropic_default_is_5(self):
        sem = get_semaphore("anthropic")
        assert sem._value == 5

    def test_openai_default_is_5(self):
        sem = get_semaphore("openai")
        assert sem._value == 5

    def test_groq_default_is_5(self):
        sem = get_semaphore("groq")
        assert sem._value == 5

    def test_ollama_default_is_1(self):
        sem = get_semaphore("ollama")
        assert sem._value == 1

    def test_unknown_provider_defaults_to_5(self):
        sem = get_semaphore("unknown_provider")
        assert sem._value == 5

    def test_user_override_respected(self):
        sem = get_semaphore("anthropic", user_override=2)
        assert sem._value == 2

    def test_user_override_beats_ollama_default(self):
        sem = get_semaphore("ollama", user_override=4)
        assert sem._value == 4

    def test_same_budget_shared_across_providers(self):
        # Single concurrency budget: providers with the same effective limit
        # share one semaphore (anthropic and openai both default to 5).
        sem1 = get_semaphore("anthropic")
        sem2 = get_semaphore("openai")
        assert sem1 is sem2

    def test_provider_switch_does_not_poison_budget(self):
        # Regression for C3: a low-limit provider (ollama=1) must not pin the
        # budget for a later higher-limit provider (anthropic=5). Before the
        # fix, the first call's limit stuck for the whole process.
        ollama_sem = get_semaphore("ollama")
        assert ollama_sem._value == 1
        anthropic_sem = get_semaphore("anthropic")
        assert anthropic_sem._value == 5
        assert anthropic_sem is not ollama_sem

    def test_reset_clears_singleton(self):
        sem1 = get_semaphore("anthropic")
        reset_semaphore()
        sem2 = get_semaphore("ollama")
        assert sem1 is not sem2
        assert sem2._value == 1

    def test_provider_defaults_table_completeness(self):
        for provider in ("anthropic", "openai", "groq", "ollama"):
            assert provider in PROVIDER_DEFAULTS

    def test_semaphore_is_asyncio_semaphore(self):
        sem = get_semaphore("anthropic")
        assert isinstance(sem, asyncio.Semaphore)
