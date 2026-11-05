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
            }
        }
        // One missing tile must be found immediately anywhere in the screen.
        foreach(int missing in new[]{0,11300,22599}) {
            int[] input = new int[22600*2]; input[missing*2]=1;
            using var tiles = new GpuBuffer<int>(ctx,input,BufferUsage.Storage | BufferUsage.CopySrc,"SparseScheduleFixture");
            Submit(ctx, encoder => scheduler.Run(encoder,tiles,22600,17));
            var result = FrameCapture.ReadBuffer(ctx,tiles);
            if(result[missing*2]!=1 || result.Where((_,i)=>i%2==0).Sum()!=1) throw new InvalidOperationException("Sparse candidate was starved.");
        }
        Console.WriteLine("[Schedule] GPU checks passed: exact budget, dispersed quadrants, sparse candidate service, non-power-of-two grids.");
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
