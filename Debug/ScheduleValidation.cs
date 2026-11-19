using System.Numerics;
using Silk.NET.WebGPU;
using SurfelGI.Core;
using SurfelGI.GiPipeline;

namespace SurfelGI.Debug;

// Read the real GPU scheduling result: catch row sweeps, budget overruns, empty
// candidate starvation, and toroidal update stripes without mirroring the permutation.
public static unsafe class ScheduleValidation
{
    public static void Run(WebGpuContext ctx, ShaderManager shaders)
    {
        using var scheduler = new SurfelSpawnScheduler(ctx, shaders);
        foreach (int count in new[] { 1, 63, 257, 22600 }) {
            for (uint frame=0;frame<3;frame++) {
                int[] input = new int[count*2];
                for(int i=0;i<count;i++) { input[i*2]=1; input[i*2+1]=i; }
                using var tiles = new GpuBuffer<int>(ctx, input, BufferUsage.Storage | BufferUsage.CopySrc, "ScheduleFixture");
                Submit(ctx, encoder => scheduler.Run(encoder,tiles,count,frame));
                int[] result = FrameCapture.ReadBuffer(ctx,tiles);
                int selected=0; int[] quadrants = new int[4];
                for(int i=0;i<count;i++) {
                    if(result[i*2+1] != i || result[i*2] is < 0 or > 1) throw new InvalidOperationException("Spawn scheduler corrupted candidate records.");
                    if(result[i*2]==1) {
                        selected++;
                        if(count==22600) quadrants[(i%200>=100?1:0)+(i/200>=56?2:0)]++;
                    }
                }
                if(selected != Math.Min(count,(int)SurfelSpawnScheduler.SpawnBudget)) throw new InvalidOperationException("Spawn scheduler lost or exceeded its budget.");
                if(count==22600 && quadrants.Any(n => n < 32)) throw new InvalidOperationException("Spawn updates are spatially concentrated.");
                var selectedTiles = FrameCapture.ReadBuffer(ctx, scheduler.SelectedTileBuffer)
                    .Take(selected).Order().ToArray();
                var flaggedTiles = Enumerable.Range(0, count).Where(tile => result[tile * 2] == 1);
                if (!selectedTiles.SequenceEqual(flaggedTiles))
                    throw new InvalidOperationException("Spawn list differs from the scheduled tile flags.");
            }
        }
        // One missing tile must be found immediately anywhere in the screen.
        foreach(int missing in new[]{0,11300,22599}) {
            int[] input = new int[22600*2]; input[missing*2]=1;
            using var tiles = new GpuBuffer<int>(ctx,input,BufferUsage.Storage | BufferUsage.CopySrc,"SparseScheduleFixture");
            Submit(ctx, encoder => scheduler.Run(encoder,tiles,22600,17));
            var result = FrameCapture.ReadBuffer(ctx,tiles);
            if(result[missing*2]!=1 || result.Where((_,i)=>i%2==0).Sum()!=1) throw new InvalidOperationException("Sparse candidate was starved.");
            var selectedTiles = FrameCapture.ReadBuffer(ctx, scheduler.SelectedTileBuffer);
            if (!selectedTiles.Where(tile => tile >= 0).SequenceEqual(new[] { missing }))
                throw new InvalidOperationException("Sparse spawn list contains a stale candidate.");
        }
        Console.WriteLine("[Schedule] GPU checks passed: exact budget, dispersed quadrants, sparse candidate service, non-power-of-two grids.");
        ValidateWorldSpaceAllocation(ctx, shaders);
    }

    private static void ValidateWorldSpaceAllocation(WebGpuContext ctx, ShaderManager shaders)
    {
        using var find = new SurfelFindMissingPass(ctx, shaders);
        using var allocate = new SurfelAllocatePass(ctx, shaders);
        using var pool = new SurfelPool(ctx, 16);
        using var prepare = new SurfelPreparePass(ctx, shaders);

        // The first two candidates overlap despite coming from distant screen tiles.
        // The others cover a corner, a parallel surface, and a separate patch.
        (int Tile, Vector3 Position, uint PackedNormal)[] candidates =
        {
            (0, Vector3.Zero, 1023u | (511u << 11) | (2047u << 21)),
            (100, new Vector3(0.005f, 0, 0), 1023u | (511u << 11) | (2047u << 21)),
            (200, Vector3.Zero, 1023u | (1023u << 11) | (1023u << 21)),
            (210, new Vector3(0, 0, 0.03f), 1023u | (511u << 11) | (2047u << 21)),
            (220, new Vector3(0.1f, 0, 0), 1023u | (511u << 11) | (2047u << 21))
        };
        const int tileCount = 257;
        var tiles = new int[tileCount * 2];
        var selected = Enumerable.Repeat(-1, tileCount).ToArray();
        for (int lane = 0; lane < candidates.Length; lane++)
        {
            var (tile, position, normal) = candidates[lane];
            tiles[tile * 2] = 1;
            tiles[tile * 2 + 1] = -1;
            selected[lane] = tile;
            find.CandidatePackedBuffer.UploadSingle(
                new Vector4(position, BitConverter.UInt32BitsToSingle(normal)), (ulong)tile);
        }
        find.TileAllocBuffer.Upload(tiles);
        find.SelectedTileBuffer.Upload(selected);
        Submit(ctx, encoder =>
        {
            prepare.Run(encoder, pool, true);
            allocate.Run(encoder, pool, find, tileCount);
        });

        var live = FrameCapture.ReadBuffer(ctx, pool.SurfelBuffer)
            .Where(surfel => surfel.Age < Constants.SURFEL_TTL).ToArray();
        if (live.Length != 4 ||
            live.Count(surfel => surfel.Normal.Z > 0.9f && surfel.PosB.X < 0.01f && surfel.PosB.Z == 0) != 1 ||
            !live.Any(surfel => surfel.Normal.Y > 0.9f) ||
            !live.Any(surfel => surfel.PosB.Z == 0.03f) ||
            !live.Any(surfel => surfel.PosB.X == 0.1f))
            throw new InvalidOperationException("World-space allocation lost a distinct surface or admitted an overlapping candidate.");
        Console.WriteLine("[Allocation] GPU checks passed: distant tile deduplication, corners, parallel surfaces, separate patches.");
    }
    private unsafe delegate void Encode(CommandEncoder* encoder);
    private static void Submit(WebGpuContext ctx, Encode encode)
    {
        var desc=new CommandEncoderDescriptor(); var encoder=ctx.Wgpu.DeviceCreateCommandEncoder(ctx.Device,in desc);
        encode(encoder);
        var cd=new CommandBufferDescriptor(); var command=ctx.Wgpu.CommandEncoderFinish(encoder,in cd);
        if(ctx.LastError!=null) throw new InvalidOperationException(ctx.LastError);
        ctx.Wgpu.QueueSubmit(ctx.Queue,1,&command); ctx.Wgpu.DeviceWaitIdle(ctx.Device);
        ctx.Wgpu.CommandBufferRelease(command); ctx.Wgpu.CommandEncoderRelease(encoder);
    }
}
