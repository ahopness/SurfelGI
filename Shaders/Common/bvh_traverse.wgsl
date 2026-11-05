const BVH_STACK_DEPTH: u32 = 60u;
const INFINITY: f32 = 1e20;
const TRI_INTERSECT_EPSILON: f32 = 1e-5;

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct BVHBoundingBox {
    min: array<f32, 3>,
    max: array<f32, 3>,
};

struct BVHNode {
    bounds: BVHBoundingBox,
    rightChildOrTriangleOffset: u32,
    splitAxisOrTriangleCount: u32,
};

struct IntersectionResult {
    didHit: bool,
    indices: vec4u,
    normal: vec3f,
    barycoord: vec3f,
    side: f32,
    dist: f32,
};

fn getVertexAttribute(barycoord: vec3f, indices: vec3u) -> vec3f {
    let n0 = bvh_attribute[indices.x].xyz;
    let n1 = bvh_attribute[indices.y].xyz;
    let n2 = bvh_attribute[indices.z].xyz;
    return barycoord.x * n0 + barycoord.y * n1 + barycoord.z * n2;
}

fn intersectsBounds(
    ray: Ray,
    bounds: BVHBoundingBox,
    invDir: vec3f,
    dist: ptr<function, f32>
) -> bool {
    let boundsMin = vec3f(bounds.min[0], bounds.min[1], bounds.min[2]);
    let boundsMax = vec3f(bounds.max[0], bounds.max[1], bounds.max[2]);

    let tMinPlane = (boundsMin - ray.origin) * invDir;
    let tMaxPlane = (boundsMax - ray.origin) * invDir;

    let tMinHit = min(tMinPlane, tMaxPlane);
    let tMaxHit = max(tMinPlane, tMaxPlane);

    let t0 = max(max(tMinHit.x, tMinHit.y), tMinHit.z);
    let t1 = min(min(tMaxHit.x, tMaxHit.y), tMaxHit.z);

    *dist = max(t0, 0.0);
    return t1 >= *dist;
}

fn intersectsTriangle(ray: Ray, a: vec3f, b: vec3f, c: vec3f) -> IntersectionResult {
    var result: IntersectionResult;
    result.didHit = false;

    let edge1 = b - a;
    let edge2 = c - a;
    let n = cross(edge1, edge2);

    let det = -dot(ray.direction, n);
    if (abs(det) < TRI_INTERSECT_EPSILON) {
        return result;
    }

    let invdet = 1.0 / det;
    let AO = ray.origin - a;
    let DAO = cross(AO, ray.direction);

    let u = dot(edge2, DAO) * invdet;
    let v = -dot(edge1, DAO) * invdet;
    let t = dot(AO, n) * invdet;
    let w = 1.0 - u - v;

    if (u < -TRI_INTERSECT_EPSILON || v < -TRI_INTERSECT_EPSILON || w < -TRI_INTERSECT_EPSILON || t < TRI_INTERSECT_EPSILON) {
        return result;
    }

    result.didHit = true;
    result.barycoord = vec3f(w, u, v);
    result.dist = t;
    result.side = sign(det);
    result.normal = result.side * normalize(n);
    return result;
}

fn intersectTriangles(offset: u32, count: u32, ray: Ray) -> IntersectionResult {
    var closestResult: IntersectionResult;
    closestResult.didHit = false;
    closestResult.dist = INFINITY;

    for (var i = offset; i < offset + count; i = i + 1u) {
        let indices = bvh_index[i].xyz;
        let a = bvh_position[indices.x].xyz;
        let b = bvh_position[indices.y].xyz;
        let c = bvh_position[indices.z].xyz;

        let triResult = intersectsTriangle(ray, a, b, c);
        if (triResult.didHit && triResult.dist < closestResult.dist) {
            closestResult = triResult;
            closestResult.indices = vec4u(indices, i);
        }
    }

    return closestResult;
}

fn bvhTrace(ray: Ray, maxDistance: f32, anyHit: bool) -> IntersectionResult {
    let invDir = sign(ray.direction) / max(abs(ray.direction), vec3f(1e-20)) + select(vec3f(0.0), vec3f(1e20), ray.direction == vec3f(0.0));
    var pointer: i32 = 0;
    var stack: array<u32, BVH_STACK_DEPTH>;
    stack[0] = 0u;

    var bestHit: IntersectionResult;
    bestHit.didHit = false;
    bestHit.dist = maxDistance;

    loop {
        if (pointer < 0 || pointer >= i32(BVH_STACK_DEPTH)) {
            break;
        }

        let currNodeIndex = stack[pointer];
        let node = bvh[currNodeIndex];
        pointer = pointer - 1;

        var boundsHitDistance: f32 = 0.0;
        if (!intersectsBounds(ray, node.bounds, invDir, &boundsHitDistance) || boundsHitDistance > bestHit.dist) {
            continue;
        }

        let boundsInfox = node.splitAxisOrTriangleCount;
        let boundsInfoy = node.rightChildOrTriangleOffset;
        let isLeaf = (boundsInfox & 0xffff0000u) != 0u;

        if (isLeaf) {
            let count = boundsInfox & 0x0000ffffu;
            let offset = boundsInfoy;
            let localHit = intersectTriangles(offset, count, ray);
            if (localHit.didHit && localHit.dist < bestHit.dist) {
                bestHit = localHit;
                if (anyHit) { return bestHit; }
            }
        } else {
            let leftIndex = currNodeIndex + 1u;
            let splitAxis = boundsInfox & 0x0000ffffu;
            let rightIndex = currNodeIndex + boundsInfoy;

            let leftToRight = ray.direction[splitAxis] >= 0.0;
            let c1 = select(rightIndex, leftIndex, leftToRight);
            let c2 = select(leftIndex, rightIndex, leftToRight);

            pointer = pointer + 1;
            stack[pointer] = c2;
            pointer = pointer + 1;
            stack[pointer] = c1;
        }
    }

    return bestHit;
}

fn bvhIntersectFirstHit(ray: Ray) -> IntersectionResult { return bvhTrace(ray, INFINITY, false); }
fn bvhOccluded(ray: Ray) -> bool { return bvhTrace(ray, INFINITY, true).didHit; }
