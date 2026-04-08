import asyncio
from benchmarks.bench_generator import BenchmarkGenerator, DifficultyLevel
from benchmarks.bench_suite import BenchmarkSuite, SchedulerMode

async def test():
    gen = BenchmarkGenerator(seed=42)
    eps = gen.generate_tier1(count=1, difficulty=DifficultyLevel.EASY)
    
    print('=== 测试 ooo_proposed ===')
    suite = BenchmarkSuite(
        scheduler_mode=SchedulerMode.OOO_PROPOSED,
        seed=42,
        max_workers=3,
        verbose=True,
        physical_delay_scale=0.01,
        ablated_dims=set(),
    )
    result = await suite.run_episode(eps[0])
    print(f'成功: {result.success}')
    print(f'OoO Promotions: {result.ooo_promotion_count}')
    print(f'错误: {result.error}')

asyncio.run(test())
