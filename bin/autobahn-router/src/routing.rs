use crate::debug_tools;
use crate::metrics;
use crate::prelude::*;
use crate::routing_objectpool::RoutingObjectPools;
use mango_feeds_connector::chain_data::AccountData;
use ordered_float::NotNan;
use router_config_lib::Config;
use router_lib::dex::SwapMode;
use router_lib::dex::{AccountProviderView, DexEdge};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::time::{Duration, Instant};
use std::u64;
use thiserror::Error;
use tracing::Level;

use crate::routing_types::*;

#[derive(Error, Debug)]
pub enum RoutingError {
    #[error("unsupported input mint {0:?}")]
    UnsupportedInputMint(Pubkey),
    #[error("unsupported output mint {0:?}")]
    UnsupportedOutputMint(Pubkey),
    #[error("no path between {0:?} and {1:?}")]
    NoPathBetweenMintPair(Pubkey, Pubkey),
    #[error("could not compute out amount")]
    CouldNotComputeOut,
}

fn best_price_paths_depth_search<F>(
    input_node: MintNodeIndex, //搜索的起始点(如:USDC)
    amount: u64,
    max_path_length: usize, //一条路径中允许的最大“跳数”或交易次数（例如，4跳）。这可以防止搜索无限进行下去，并使路径保持相对简单
    max_accounts: usize,    //允许的accounts最大数
    //当前整个图的数据结构
    //out_edges_per_node是一个二维数组!!!
    //这里的T就是Vec<EdgeWithNodes>
    //out_edges_per_node就是一个邻接表
    //第一层(索引层)代表图的一个顶点,也就是一条边
    //第二层是Vec<EdgeWithNodes>代表他的每一条边(out的意思是只包含从当前节点指向其他节点的边)
    out_edges_per_node: &MintVec<Vec<EdgeWithNodes>>,

    // 背景: 这种寻路函数在交易聚合器中会被极度频繁地调用。如果在每次调用时都在函数内部创建巨
    //       大的数据结构来存储中间结果和最终结果，那么内存分配和释放的开销会成为严重的性能瓶颈。
    // 解决方案: 函数要求调用者预先分配 (pre-allocate) 好内存，然后通过可变引用 &mut 传入。
    //          函数在这些已经存在的内存上进行读写，实现了“零分配”或近乎“零分配”的执行，这对于热点路径代码至关重要。
    //第71行和第76行的 assert! 检查就是为了确保这些缓冲区的大小是正确的。

    // 这是一个预先分配好内存的、用来按节点存放最佳路径的数据结构
    // 这里和out_edges_per_node不同的地方组要是最内层的结构(NotNan<f64>, Vec<EdgeWithNodes>)
    // 他多了一个NotNan<f64>,代表着路径的最终价值/权重/兑换率。比如，从 A 到 D 的
    // 一条路径，最终能将 1 个 A 兑换成 1.05 个 D，这个 f64 就可能是 1.05
    // Vec<EdgeWithNodes>: 得到这个最终价值所需要经过的具体路径，也就是一个边的有序列表。比如 [A->B的边, B->C的边, C->D的边]。
    // 这里装的是从起始节点到目标节点的最佳路径
    // 起始节点: input_node (比如 USDC)
    // 目标节点: best_paths_by_node_prealloc 的索引 (比如 USDT、SOL、DAI 等)
    best_paths_by_node_prealloc: &mut MintVec<Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>>,

    // 这是一个预分配的、可变的、以节点为索引的数组，用于在寻路算法
    // （比如贝尔曼-福特或 Dijkstra 算法的变种）执行期间，实时记录到达每个节点的前 3 条最优路径的收益
    // 这个包含 3 个浮点数的数组，存储的是从起始节点(入参的input_node)最佳路径的最终收益额。
    // 比如，它可能存储着 [1.05, 1.02, 1.01]，代表到达这个节点最好的三条路径分别能让初始资金翻 1.05 倍、1.02 倍和 1.01 倍
    // 同样:他和best_paths_by_node_prealloc一样
    // 起始节点: input_node (比如 USDC)
    // 目标节点: best_paths_by_node_prealloc 的索引 (比如 USDT、SOL、DAI 等)
    best_by_node_prealloc: &mut Vec<BestVec3>,

    // best_paths_by_node_prealloc和best_by_node_prealloc的区别
    // best_by_node_prealloc (当前这个): 存储的是中间状态。它只关心从初始节点到达某个节点的收益数字 (f64)，
    // 不关心具体是怎么走过来的。这使得算法在迭代时可以快速比较收益值，决定下一步要探索哪个节点。它更轻量，用于算法的核心循环。
    // best_paths_by_node_prealloc (上一个): 存储的是最终结果。它不仅包含收益 (f64)，还包含了完整的路径 (Vec<EdgeWithNodes>)。
    // 它更重量级，是在找到一条更好的路径后，用来记录完整解的地方。

    //这是一个函数（技术上讲是闭包），用于计算单次交易的结果。它作为参数被传递进来，
    //这使得搜索算法本身变得非常通用，并与 ExactIn 和 ExactOut 的具体定价逻辑解耦
    edge_price: F,
    hot_mints: &HashSet<MintNodeIndex>,
    avoid_cold_mints: bool,
    swap_mode: SwapMode,
) -> anyhow::Result<MintVec<Vec<(f64, Vec<EdgeWithNodes>)>>>
where
    F: Fn(EdgeIndex, u64) -> Option<EdgeInfo>,
{
    debug!(input_node = input_node.idx_raw(), amount, "finding path");

    //强制max_accounts不超过40
    let max_accounts = max_accounts.min(40);

    if tracing::event_enabled!(Level::TRACE) {
        let count = count_edges(out_edges_per_node);
        trace!(count, "available edges");
        trace!(
            count = out_edges_per_node[0.into()].len(),
            "available edges out of 0"
        );
    };

    // 寻路算法（如我们之前讨论的）会填满 best_by_node_prealloc 和 best_paths_by_node_prealloc
    // 这两个“记分板”和“日志”。此时，我们已经知道了到达每个终点的最佳收益，以及到达每个中间节点的前驱节点是什么。
    // 当算法需要将一条最优路径（比如从A到D的最佳路径）构造成一个明确的、有序的边列表时，它就会使用这个 path 变量。
    // 回溯构建路径: 当算法需要将一条最优路径（比如从A到D的最佳路径）构造成一个明确的、有序的边列表时，它就会使用这个 path 变量。
    // 它会从终点 D 开始。
    // 找到到达 D 的最佳路径是从哪个前驱节点 C 来的，然后把 C -> D 这条边加入到 path 中。
    // 接着，它跳到节点 C，看它是从哪个前驱节点 B 来的，然后把 B -> C 这条边加入到 path 中。
    // 如此反复，直到回溯到起点 A。
    // 此时，path 中可能存储着 [C->D, B->C, A->B]。
    // 反转和使用: 因为是倒着回溯的，所以需要将 path 反转（reverse）才能得到正确的顺序：[A->B, B->C, C->D]。然后就可以把这条构建好的路径存到最终的结果里，或者直接返回。
    // 清空和复用: 在为下一条路径进行回溯构建之前，这个 path 会被清空 (path.clear())，而不是重新分配内存。这样，在同一个函数调用中，它可以被反复用来构建多条不同的路径，从而避免了不必要的内存分配，这也是一种性能优化。
    let mut path = Vec::new();

    let mut best_paths_by_node: &mut MintVec<Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>> =
        best_paths_by_node_prealloc;

        //这里说明
    assert_eq!(
        best_by_node_prealloc.len(),
        8 * out_edges_per_node.len(),
        "best_by_node_prealloc len error"
    );
    assert!(
        best_by_node_prealloc.iter().all(|v| v.len() == 3),
        "best_by_node_prealloc vector items length error"
    );

    let mut best_by_node = best_by_node_prealloc;

    let mut stats = vec![0; 2];

    // 使用直接路径初始化 best_paths_by_node 和 best_by_node
    {
        let current_account_count = 0;
        let in_amount = amount as f64;
        //这里的out_edge就是当前输入节点的每条边
        for out_edge in &out_edges_per_node[input_node] {
            try_append_to_best_results(
                &mut best_paths_by_node,
                &mut path,
                &mut best_by_node,
                &mut stats,
                in_amount,
                max_accounts,
                current_account_count,
                &edge_price,
                &out_edge,
                swap_mode,
            );
        }
    }

    // 深度优先搜索
    walk(
        &mut best_paths_by_node,
        &mut path,
        &mut best_by_node,
        &mut stats,
        amount as f64,
        input_node,
        max_path_length,
        max_accounts,
        0,
        out_edges_per_node,
        &edge_price,
        hot_mints,
        avoid_cold_mints,
        swap_mode,
    );

    let good_paths = best_paths_by_node
        .iter()
        .map(|best_paths| {
            best_paths
                .into_iter()
                .filter_map(|(out, edges)| {
                    // 没能成功 .iter_into()
                    let edges = edges.clone();
                    match swap_mode {
                        SwapMode::ExactIn => {
                            out.is_sign_positive().then_some((out.into_inner(), edges))
                        }
                        SwapMode::ExactOut => out.is_finite().then_some((out.into_inner(), edges)),
                    }
                })
                .collect_vec()
        })
        .collect_vec();

    debug!(
        input_node = input_node.idx_raw(),
        amount,
        probed = stats[0],
        skipped = stats[1],
        "done"
    );

    Ok(good_paths.into())
}

fn walk<F2>(
    best_paths_by_node: &mut MintVec<Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>>,
    path: &mut Vec<EdgeWithNodes>,
    // 按 "bucket" 索引，既不是节点索引也不是边索引
    best_by_node: &mut Vec<BestVec3>,
    stats: &mut Vec<u64>,
    in_amount: f64,
    input_node: MintNodeIndex,
    max_path_length: usize,
    max_accounts: usize,
    current_account_count: usize,
    out_edges_per_node: &MintVec<Vec<EdgeWithNodes>>,
    edge_price_fn: &F2,
    hot_mints: &HashSet<MintNodeIndex>,
    avoid_cold_mints: bool,
    swap_mode: SwapMode,
) where
    F2: Fn(EdgeIndex, u64) -> Option<EdgeInfo>,
{
    if max_path_length == 0
        || max_accounts < (current_account_count + 4)
        || in_amount.is_nan()
        || in_amount.is_sign_negative()
    {
        return;
    }

    stats[0] += 1;

    for out_edge in &out_edges_per_node[input_node] {
        // 禁止循环
        if path
            .iter()
            .any(|step| step.source_node == out_edge.target_node)
        {
            continue;
        }

        let Some((edge_info, out_amount)) = try_append_to_best_results(
            best_paths_by_node,
            path,
            best_by_node,
            stats,
            in_amount,
            max_accounts,
            current_account_count,
            edge_price_fn,
            &out_edge,
            swap_mode,
        ) else {
            continue;
        };

        // 遇到冷门 mint 时停止深度搜索
        if avoid_cold_mints && hot_mints.len() > 0 && !hot_mints.contains(&out_edge.source_node) {
            stats[1] += 1;
            continue;
        }

        path.push(out_edge.clone());

        walk(
            best_paths_by_node,
            path,
            best_by_node,
            stats,
            out_amount,
            out_edge.target_node,
            max_path_length - 1,
            max_accounts,
            current_account_count + edge_info.accounts,
            out_edges_per_node,
            edge_price_fn,
            hot_mints,
            avoid_cold_mints,
            swap_mode,
        );

        //dfs的回溯
        path.pop();
    }
}

//当算法考虑走一条特定的边时，计算结果并决定是否要更新我们的最佳路径记录
fn try_append_to_best_results<F2>(
    best_paths_by_node: &mut MintVec<Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>>,
    path: &Vec<EdgeWithNodes>,
    best_by_node: &mut Vec<BestVec3>,
    stats: &mut Vec<u64>,
    in_amount: f64,
    max_accounts: usize,
    current_account_count: usize,
    edge_price_fn: &F2,
    out_edge: &EdgeWithNodes,
    swap_mode: SwapMode,
) -> Option<(EdgeInfo, f64)>
where
    F2: Fn(EdgeIndex, u64) -> Option<EdgeInfo>,
{
    //调用传入的定价函数 edge_price_fn，询问："如果我用 in_amount 的代币去走 out_edge 这条边，能得到什么结果？"
    // 如果这条边无效（比如流动性不足、池子关闭等），定价函数会返回 None，函数直接退出。
    // 如果有效，会返回 EdgeInfo，包含了兑换比率 (price) 和需要的账户数 (accounts)。
    let Some(edge_info) = edge_price_fn(out_edge.edge, in_amount as u64) else {
        return None;
    };
    // 检查账户数约束。Solana 交易有账户数上限，如果走这条边会导致总账户数超过 max_accounts，就放弃这条路径。
    if current_account_count + edge_info.accounts > max_accounts {
        return None;
    }

    // 计算走这条边后能得到的输出金额：输入金额 × 兑换比率 = 输出金额
    let out_amount = in_amount * edge_info.price;

    //从 best_paths_by_node 中取出到达目标节点 (out_edge.target_node) 的最佳路径列表。这个列表按收益排序，存储着到达该节点的前 N 条最优路径
    let best_paths = &mut best_paths_by_node[out_edge.target_node];

    // 拿到从初始节点到out_edge.target_node的所有路径中，收益最差的那条路径的收益值
    let worst = best_paths
        .last()
        .map(|(p, _)| p.into_inner())
        .unwrap_or(match swap_mode {
            SwapMode::ExactIn => f64::NEG_INFINITY,
            SwapMode::ExactOut => f64::INFINITY,
        });

    if (swap_mode == SwapMode::ExactOut && out_amount < worst)
    //用户指定了确切的输入数量
    //要最大化输出收益
        || (swap_mode == SwapMode::ExactIn && out_amount > worst)
    {
        // 如果条件满足（新路径更好），就调用 replace_worst_path 函数：
        // 把最差的那条路径从列表中移除
        // 把当前这条新的更好的路径加入列表
        // 重新排序，维持"最佳路径列表"的有序性
        replace_worst_path(path, out_edge, out_amount, best_paths, swap_mode);
    }

    let account_bucket = (current_account_count + edge_info.accounts).div_euclid(8);
    let target_node_bucket = out_edge.target_node.idx_raw() * 8 + account_bucket as u32;
    let target_node_bucket = target_node_bucket as usize;
    let values = &best_by_node[target_node_bucket];
    let smaller_value_index = get_already_seen_worst_kept_output(values);

    if values[smaller_value_index] > out_amount {
        stats[1] += 1;
        return None;
    }

    best_by_node[target_node_bucket][smaller_value_index] = out_amount;
    Some((edge_info, out_amount))
}

fn get_already_seen_worst_kept_output(values: &BestVec3) -> usize {
    values
        .iter()
        .position_min_by_key(|v| v.floor() as u64)
        .expect("can't find min index")
}

fn replace_worst_path(
    current_path: &Vec<EdgeWithNodes>,
    added_hop: &EdgeWithNodes,
    out_amount: f64,
    best_paths: &mut Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>,
    swap_mode: SwapMode,
) {
    // TODO 性能不佳 - 尝试寻找比这更好的解决方案
    let already_exists = path_already_in(current_path, added_hop, out_amount, best_paths);

    if !already_exists {
        let mut added_path = current_path.clone();
        added_path.push(added_hop.clone());

        best_paths.pop();
        best_paths.push((NotNan::new(out_amount).unwrap(), added_path));
        match swap_mode {
            SwapMode::ExactIn => best_paths.sort_by_key(|(p, _)| std::cmp::Reverse(*p)),
            SwapMode::ExactOut => best_paths.sort_by_key(|(p, _)| *p),
        };
    }
}

fn path_already_in(
    current_path: &Vec<EdgeWithNodes>,
    added_hop: &EdgeWithNodes,
    out_amount: f64,
    best_paths: &Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>,
) -> bool {
    let mut already_exists = false;
    'outer: for existing_path in best_paths.iter() {
        if existing_path.0 != NotNan::new(out_amount).unwrap() {
            continue;
        }
        if existing_path.1.len() != (current_path.len() + 1) {
            continue;
        }

        for i in 0..current_path.len() {
            if existing_path.1[i].edge != current_path[i].edge {
                continue 'outer;
            }
        }

        if existing_path.1[existing_path.1.len() - 1].edge != added_hop.edge {
            continue;
        }

        already_exists = true;
        break;
    }
    already_exists
}

fn count_edges(edges: &[Vec<EdgeWithNodes>]) -> u64 {
    let mut set = HashSet::new();
    for node in 0..edges.len() {
        for edge in &edges[node] {
            set.insert(edge.edge);
        }
    }
    return set.len() as u64;
}

struct PathDiscoveryCacheEntry {
    timestamp_millis: u64,
    in_amount: f64,
    max_account: usize,
    // mint x mint
    edges: Vec<Vec<EdgeIndex>>,
}

// 实现了路径缓存的逻辑
struct PathDiscoveryCache {
    // 一个哈希表，存储着从A代币到B代币的已发现路径
    cache: HashMap<(MintNodeIndex, MintNodeIndex, SwapMode), Vec<PathDiscoveryCacheEntry>>,
    last_expire_timestamp_millis: u64,
    max_age_millis: u64,
}

// 提供对缓存的增、查、删（失效）等操作
impl PathDiscoveryCache {
    fn expire_old(&mut self) {
        let now = millis_since_epoch();
        if now - self.last_expire_timestamp_millis < 1000 {
            return;
        }
        self.cache.retain(|_k, entries| {
            entries.retain(|entry| {
                entry.timestamp_millis > now || now - entry.timestamp_millis < self.max_age_millis
            });
            !entries.is_empty()
        });
        self.last_expire_timestamp_millis = now;
    }

    fn expire_old_by_key(max_age_millis: u64, entries: &mut Vec<PathDiscoveryCacheEntry>) {
        let now = millis_since_epoch();
        entries.retain(|entry| {
            entry.timestamp_millis > now || now - entry.timestamp_millis < max_age_millis
        });
    }

    fn insert(
        &mut self,
        from: MintNodeIndex,
        to: MintNodeIndex,
        swap_mode: SwapMode,
        in_amount: u64,
        max_accounts: usize,
        timestamp_millis: u64,
        mut edges: Vec<Vec<EdgeIndex>>,
    ) {
        // 大约 3-4
        trace!(
            "insert entry into discovery cache with edges cardinality of {}",
            edges.len()
        );

        let max_accounts_bucket = Self::compute_account_bucket(max_accounts);
        let entry = self.cache.entry((from, to, swap_mode)).or_default();

        // 尝试减少内存占用...
        for path in &mut edges {
            path.shrink_to_fit();
        }
        edges.shrink_to_fit();

        let new_elem = PathDiscoveryCacheEntry {
            timestamp_millis,
            in_amount: in_amount as f64,
            max_account: max_accounts_bucket,
            edges,
        };

        let pos = entry
            .binary_search_by_key(&(in_amount, max_accounts_bucket), |x| {
                (x.in_amount.round() as u64, x.max_account)
            })
            .unwrap_or_else(|e| e);

        // 如果已存在则替换（而不是因为旧路径和新路径使缓存大小加倍）
        if pos < entry.len()
            && entry[pos].max_account == max_accounts_bucket
            && entry[pos].in_amount.round() as u64 == in_amount
        {
            entry[pos] = new_elem;
            return;
        }

        entry.insert(pos, new_elem);
    }

    fn get(
        &mut self,
        from: MintNodeIndex,
        to: MintNodeIndex,
        swap_mode: SwapMode,
        in_amount: u64,
        max_accounts: usize,
    ) -> (Option<&Vec<Vec<EdgeIndex>>>, Option<&Vec<Vec<EdgeIndex>>>) {
        let in_amount = in_amount as f64;
        let max_accounts_bucket = Self::compute_account_bucket(max_accounts);
        let Some(entries) = self.cache.get_mut(&(from, to, swap_mode)) else {
            // 缓存未命中
            metrics::PATH_DISCOVERY_CACHE_MISSES.inc();
            return (None, None);
        };
        metrics::PATH_DISCOVERY_CACHE_HITS.inc();

        Self::expire_old_by_key(self.max_age_millis, entries);

        let (mut lower, mut upper) = (None, None);
        for entry in entries {
            if entry.max_account != max_accounts_bucket {
                continue;
            }
            if entry.in_amount <= in_amount {
                lower = Some(&entry.edges);
            }
            if entry.in_amount > in_amount {
                upper = Some(&entry.edges);
                break;
            }
        }

        (lower, upper)
    }

    fn invalidate(&mut self, from: MintNodeIndex, to: MintNodeIndex, max_accounts: usize) {
        let max_accounts_bucket = Self::compute_account_bucket(max_accounts);
        let Some(entries) = self.cache.get_mut(&(from, to, SwapMode::ExactIn)) else {
            return;
        };

        entries.retain(|x| x.max_account != max_accounts_bucket);

        let Some(entries) = self.cache.get_mut(&(from, to, SwapMode::ExactOut)) else {
            return;
        };
        entries.retain(|x| x.max_account != max_accounts_bucket);
    }

    fn compute_account_bucket(max_accounts: usize) -> usize {
        max_accounts.div_euclid(5)
    }
}

/// global singleton to manage routing
#[allow(dead_code)]
pub struct Routing {
    // 按 EdgeIndex 索引
    edges: Vec<Arc<Edge>>,

    // 按 MintNodeIndex 索引
    mints: MintVec<Pubkey>,

    // 用于 best_price_paths_depth_search 的内存池
    objectpools: RoutingObjectPools,

    // 寻路准备
    // mint pubkey -> NodeIndex
    mint_to_index: HashMap<Pubkey, MintNodeIndex>,

    path_discovery_cache: RwLock<PathDiscoveryCache>,

    // 保留修剪后的边一段时间（加速搜索）
    // 第一个用于 exact in，第二个用于 exact out
    pruned_out_edges_per_mint_index_exact_in: RwLock<(Instant, MintVec<Vec<EdgeWithNodes>>)>,
    pruned_out_edges_per_mint_index_exact_out: RwLock<(Instant, MintVec<Vec<EdgeWithNodes>>)>,
    path_warming_amounts: Vec<u64>,

    // 优化和启发式算法
    overquote: f64,
    max_path_length: usize,
    retain_path_count: usize,
    max_edge_per_pair: usize,
    max_edge_per_cold_pair: usize,
}

impl Routing {
    pub fn new(
        configuration: &Config,
        path_warming_amounts: Vec<u64>,
        edges: Vec<Arc<Edge>>,
    ) -> Self {
        let mints: MintVec<Pubkey> = edges
            .iter()
            .flat_map(|e| [e.input_mint, e.output_mint])
            .unique()
            .collect_vec()
            .into();
        let mint_to_index: HashMap<Pubkey, MintNodeIndex> = mints
            .iter()
            .enumerate()
            .map(|(i, mint_pubkey)| (*mint_pubkey, i.into()))
            .collect();

        info!(
            "Setup routing algorithm with {} edges and {} distinct mints",
            edges.len(),
            mints.len()
        );

        let mint_count = mints.len();
        let retain_path_count = configuration.routing.retain_path_count.unwrap_or(10);

        Self {
            edges,
            mints,
            objectpools: RoutingObjectPools::new(mint_count, retain_path_count),
            mint_to_index,
            path_discovery_cache: RwLock::new(PathDiscoveryCache {
                cache: Default::default(),
                last_expire_timestamp_millis: 0,
                max_age_millis: configuration.routing.path_cache_validity_ms,
            }),
            pruned_out_edges_per_mint_index_exact_in: RwLock::new((
                Instant::now() - Duration::from_secs(3600 * 24),
                MintVec::new_from_prototype(0, vec![]),
            )),
            pruned_out_edges_per_mint_index_exact_out: RwLock::new((
                Instant::now() - Duration::from_secs(3600 * 24),
                MintVec::new_from_prototype(0, vec![]),
            )),
            path_warming_amounts,
            overquote: configuration.routing.overquote.unwrap_or(0.20),
            max_path_length: configuration.routing.max_path_length.unwrap_or(4),
            retain_path_count,
            max_edge_per_pair: configuration.routing.max_edge_per_pair.unwrap_or(8),
            max_edge_per_cold_pair: configuration.routing.max_edge_per_cold_pair.unwrap_or(3),
        }
    }

    /// 如果启用了路径预热，这里应该什么都不做
    pub fn prepare_pruned_edges_if_not_initialized(
        &self,
        hot_mints: &HashSet<Pubkey>,
        swap_mode: SwapMode,
    ) {
        let reader = match swap_mode {
            SwapMode::ExactIn => self
                .pruned_out_edges_per_mint_index_exact_in
                .read()
                .unwrap(),
            SwapMode::ExactOut => self
                .pruned_out_edges_per_mint_index_exact_out
                .read()
                .unwrap(),
        };

        let need_refresh = reader.1.len() == 0 || reader.0.elapsed() > Duration::from_secs(60 * 15);
        drop(reader);
        if need_refresh {
            self.prepare_pruned_edges_and_cleanup_cache(hot_mints, swap_mode);
        }
    }

    // 根据条件匹配每个请求调用
    #[tracing::instrument(skip_all, level = "trace")]
    pub fn prepare_pruned_edges_and_cleanup_cache(
        &self,
        hot_mints: &HashSet<Pubkey>,
        swap_mode: SwapMode,
    ) {
        debug!("prepare_pruned_edges_and_cleanup_cache started");
        self.path_discovery_cache.write().unwrap().expire_old();

        let (valid_edge_count, out_edges_per_mint_index) = Self::select_best_pools(
            &hot_mints,
            self.max_edge_per_pair,
            self.max_edge_per_cold_pair,
            &self.path_warming_amounts,
            &self.edges,
            &self.mint_to_index,
            swap_mode,
        );

        let mut writer = match swap_mode {
            SwapMode::ExactIn => self
                .pruned_out_edges_per_mint_index_exact_in
                .write()
                .unwrap(),
            SwapMode::ExactOut => self
                .pruned_out_edges_per_mint_index_exact_out
                .write()
                .unwrap(),
        };
        if valid_edge_count > 0 {
            (*writer).0 = Instant::now();
        }
        if !(*writer).1.try_clone_from(&out_edges_per_mint_index) {
            debug!("Failed to use clone_into out_edges_per_mint_index, falling back to slower assignment");
            (*writer).1 = out_edges_per_mint_index;
        }

        debug!("prepare_pruned_edges_and_cleanup_cache done");
    }

    fn compute_price_impact(edge: &Arc<Edge>) -> Option<f64> {
        let state = edge.state.read().unwrap();
        if !state.is_valid() || state.cached_prices.len() < 2 {
            return None;
        }

        let first = state.cached_prices[0].1;
        let last = state.cached_prices[state.cached_prices.len() - 1].1;

        if first == 0.0 || last == 0.0 {
            return None;
        }

        if first < last {
            debug!(
                edge = edge.id.desc(),
                input = debug_tools::name(&edge.id.input_mint()),
                output = debug_tools::name(&edge.id.output_mint()),
                "weird thing happening"
            );

            // 奇怪，大额交易的价格应该比小额的差
            // 但在 openbook 上因为 lot size 的原因可能会发生这种情况
            Some(last / first - 1.0)
        } else {
            Some(first / last - 1.0)
        }
    }

    fn select_best_pools(
        hot_mints: &HashSet<Pubkey>,
        max_count_for_hot: usize,
        max_edge_per_cold_pair: usize,
        path_warming_amounts: &Vec<u64>,
        all_edges: &Vec<Arc<Edge>>,
        mint_to_index: &HashMap<Pubkey, MintNodeIndex>,
        swap_mode: SwapMode,
    ) -> (i32, MintVec<Vec<EdgeWithNodes>>) {
        let mut result = HashSet::new();

        // 理论上，对于 exact in 最好的，对于 exact out 也应该是最好的
        for (i, _amount) in path_warming_amounts.iter().enumerate() {
            let mut best = HashMap::<(Pubkey, Pubkey), Vec<(EdgeIndex, f64)>>::new();

            for (edge_index, edge) in all_edges.iter().enumerate() {
                if swap_mode == SwapMode::ExactOut && !edge.supports_exact_out() {
                    continue;
                }

                let edge_index: EdgeIndex = edge_index.into();
                let state = edge.state.read().unwrap();
                if !state.is_valid() || state.cached_prices.len() < i {
                    continue;
                }
                let price = state.cached_prices[i].1;
                if price.is_nan() || price.is_sign_negative() || price <= 0.000000001 {
                    continue;
                }

                let entry = best.entry((edge.input_mint, edge.output_mint));
                let is_hot = hot_mints.is_empty()
                    || (hot_mints.contains(&edge.input_mint)
                        && hot_mints.contains(&edge.output_mint));
                let max_count = if is_hot {
                    max_count_for_hot
                } else {
                    max_edge_per_cold_pair
                };

                match entry {
                    Occupied(mut e) => {
                        let vec = e.get_mut();
                        if vec.len() == max_count {
                            let should_replace = vec[max_count - 1].1 < price;
                            if should_replace {
                                vec[max_count - 1] = (edge_index, price);
                                vec.sort_unstable_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                            }
                        } else {
                            vec.push((edge_index, price));
                            vec.sort_unstable_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                            vec.truncate(max_count);
                        }
                    }
                    Vacant(v) => {
                        v.insert(vec![(edge_index, price)]);
                    }
                }
            }

            let res: Vec<EdgeIndex> = best
                .into_iter()
                .map(|x| x.1)
                .flatten()
                .map(|x| x.0)
                .collect();
            result.extend(res);
        }

        let mut valid_edge_count = 0;
        // TODO 如何重用？或许可以用对象池
        let mut out_edges_per_mint_index: MintVec<Vec<EdgeWithNodes>> =
            MintVec::new_from_prototype(mint_to_index.len(), vec![]);
        let mut lower_price_impact_edge_for_mint_and_direction =
            HashMap::<(MintNodeIndex, bool), (f64, EdgeIndex)>::new();
        let mut has_edge_for_mint = HashSet::new();

        let mut skipped_bad_price_impact = 0;

        for edge_index in result.iter() {
            let edge = &all_edges[edge_index.idx()];
            let in_index = mint_to_index[&edge.input_mint];
            let out_index = mint_to_index[&edge.output_mint];
            let in_key = (in_index, true);
            let out_key = (out_index, false);

            let price_impact = Self::compute_price_impact(&edge).unwrap_or(9999.9999);
            Self::update_lowest_price_impact(
                &mut lower_price_impact_edge_for_mint_and_direction,
                *edge_index,
                in_key,
                price_impact,
            );
            Self::update_lowest_price_impact(
                &mut lower_price_impact_edge_for_mint_and_direction,
                *edge_index,
                out_key,
                price_impact,
            );

            if price_impact > 0.25 {
                skipped_bad_price_impact += 1;
                continue;
            }

            match swap_mode {
                SwapMode::ExactIn => {
                    out_edges_per_mint_index[in_index].push(EdgeWithNodes {
                        source_node: in_index,
                        target_node: out_index,
                        edge: *edge_index,
                    });
                }
                SwapMode::ExactOut => {
                    out_edges_per_mint_index[out_index].push(EdgeWithNodes {
                        source_node: out_index,
                        target_node: in_index,
                        edge: *edge_index,
                    });
                }
            }

            has_edge_for_mint.insert(in_key);
            has_edge_for_mint.insert(out_key);
            valid_edge_count += 1;
        }

        for (key, (_, edge_index)) in lower_price_impact_edge_for_mint_and_direction {
            let has_edge_for_mint_and_direction = has_edge_for_mint.contains(&key);
            if has_edge_for_mint_and_direction {
                continue;
            }
            let edge = &all_edges[edge_index.idx()];
            let in_index = mint_to_index[&edge.input_mint];
            let out_index = mint_to_index[&edge.output_mint];
            let in_key = (in_index, true);
            let out_key = (out_index, false);

            match swap_mode {
                SwapMode::ExactIn => {
                    out_edges_per_mint_index[in_index].push(EdgeWithNodes {
                        source_node: in_index,
                        target_node: out_index,
                        edge: edge_index,
                    });
                }
                SwapMode::ExactOut => {
                    out_edges_per_mint_index[out_index].push(EdgeWithNodes {
                        source_node: out_index,
                        target_node: in_index,
                        edge: edge_index,
                    });
                }
            }

            has_edge_for_mint.insert(in_key);
            has_edge_for_mint.insert(out_key);
            valid_edge_count += 1;
            skipped_bad_price_impact -= 1;
        }

        if valid_edge_count > 0 {
            warn!(valid_edge_count, skipped_bad_price_impact, "pruning");
        }

        // for mint_vec in out_edges_per_mint_index.iter() {
        //     for mint in mint_vec {
        //         let input_mint = mint_to_index.iter().filter(|(_, x)| **x==mint.source_node).map(|(pk,_)| *pk).collect_vec();
        //         let output_mint = mint_to_index.iter().filter(|(_, x)| **x==mint.target_node).map(|(pk,_)| *pk).collect_vec();
        //         info!("input_mint {:?} {:?}", input_mint, output_mint );
        //     }
        // }

        (valid_edge_count, out_edges_per_mint_index)
    }

    fn update_lowest_price_impact(
        lower_price_impact_edge_for_mint_and_direction: &mut HashMap<
            (MintNodeIndex, bool), // (mint 索引, 该 mint 是否为边的'输入' mint)
            (f64, EdgeIndex),
        >,
        edge_index: EdgeIndex,
        key: (MintNodeIndex, bool),
        price_impact: f64,
    ) {
        match lower_price_impact_edge_for_mint_and_direction.entry(key) {
            Occupied(mut e) => {
                if e.get().0 > price_impact {
                    e.insert((price_impact, edge_index.clone()));
                }
            }
            Vacant(e) => {
                e.insert((price_impact, edge_index.clone()));
            }
        }
    }

    // 每个请求调用一次
    #[tracing::instrument(skip_all, level = "trace")]
    fn lookup_edge_index_paths<'a>(
        &self,
        paths: impl Iterator<Item = &'a Vec<EdgeIndex>>,
    ) -> Vec<Vec<Arc<Edge>>> {
        paths
            .map(|path| {
                path.iter()
                    .map(|&edge_index| self.edges[edge_index.idx()].clone())
                    .collect_vec()
            })
            .collect_vec()
    }

    fn edge_info(&self, edge_index: EdgeIndex, _now_ms: u64, in_amount: u64) -> Option<EdgeInfo> {
        let edge = &self.edges[edge_index.idx()];
        let price = edge
            .state
            .read()
            .unwrap()
            .cached_price_for(in_amount)
            .map(|(price, _ln_price)| price)?;

        Some(EdgeInfo {
            price,
            accounts: edge.accounts_needed,
        })
    }

    fn edge_info_exact_out(
        &self,
        edge_index: EdgeIndex,
        _now_ms: u64,
        amount: u64,
    ) -> Option<EdgeInfo> {
        let edge = &self.edges[edge_index.idx()];
        let price = edge
            .state
            .read()
            .unwrap()
            .cached_price_exact_out_for(amount)
            .map(|(price, _ln_price)| price)?;
        Some(EdgeInfo {
            price,
            accounts: edge.accounts_needed,
        })
    }

    pub fn prepare_cache_for_input_mint<F>(
        &self,
        input_mint: &Pubkey,
        in_amount: u64,
        max_accounts: usize,
        filter: F,
    ) -> anyhow::Result<()>
    where
        F: Fn(&Pubkey, &Pubkey) -> bool,
    {
        // 签名者 + autobahn-executor程序 + token程序 + 源代币账户（其他的在交易边里）
        // + ATA程序 + 系统程序 + mint
        let min_accounts_needed = 7;

        let Some(&input_index) = self.mint_to_index.get(input_mint) else {
            bail!("unsupported input mint {input_mint}"); // TODO
        };

        let timestamp = millis_since_epoch();

        let new_paths_by_out_node = self.generate_best_paths(
            in_amount,
            timestamp,
            max_accounts,
            min_accounts_needed,
            input_index,
            &self
                .pruned_out_edges_per_mint_index_exact_in
                .read()
                .unwrap()
                .1,
            &HashSet::new(),
            false,
            self.max_path_length,
        )?;

        let new_paths_by_out_node_exact_out = self.generate_best_paths_exact_out(
            in_amount,
            timestamp,
            max_accounts,
            min_accounts_needed,
            input_index,
            &self
                .pruned_out_edges_per_mint_index_exact_out
                .read()
                .unwrap()
                .1,
            &HashSet::new(),
            false,
            self.max_path_length,
        )?;

        let mut writer = self.path_discovery_cache.write().unwrap();
        for (out_index, new_paths) in new_paths_by_out_node.into_iter().enumerate() {
            let out_index: MintNodeIndex = out_index.into();
            if !filter(&input_mint, &self.mints[out_index]) {
                continue;
            }
            writer.insert(
                input_index,
                out_index,
                SwapMode::ExactIn,
                in_amount,
                max_accounts,
                timestamp,
                new_paths,
            );
        }

        for (out_index, new_paths) in new_paths_by_out_node_exact_out.into_iter().enumerate() {
            let out_index: MintNodeIndex = out_index.into();
            if !filter(&input_mint, &self.mints[out_index]) {
                continue;
            }
            writer.insert(
                input_index,
                out_index,
                SwapMode::ExactOut,
                in_amount,
                max_accounts,
                timestamp,
                new_paths,
            );
        }

        Ok(())
    }

    fn prepare(
        s: &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
        e: &Arc<Edge>,
        c: &AccountProviderView,
    ) -> Option<Arc<dyn DexEdge>> {
        s.entry(e.unique_id())
            .or_insert_with(move || e.prepare(c).ok())
            .clone()
    }

    fn compute_out_amount_from_path(
        chain_data: &AccountProviderView,
        snap: &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
        path: &[Arc<Edge>],
        amount: u64,
        add_cooldown: bool,
    ) -> anyhow::Result<Option<(u64, u64)>> /* (报价, 缓存价) */ {
        let mut current_in_amount = amount;
        let mut current_in_amount_dumb = amount;
        let prepare = Self::prepare;

        for edge in path {
            if !edge.state.read().unwrap().is_valid() {
                warn!(edge = edge.desc(), "invalid edge");
                return Ok(None);
            }
            let prepared_quote = match prepare(snap, edge, chain_data) {
                Some(p) => p,
                _ => bail!(RoutingError::CouldNotComputeOut),
            };
            let quote_res = edge.quote(&prepared_quote, chain_data, current_in_amount);
            let Ok(quote) = quote_res else {
                if add_cooldown {
                    edge.state
                        .write()
                        .unwrap()
                        .add_cooldown(&Duration::from_secs(30));
                }
                warn!(
                    edge = edge.desc(),
                    amount,
                    "failed to quote, err: {:?}",
                    quote_res.unwrap_err()
                );
                return Ok(None);
            };

            if quote.out_amount == 0 {
                if add_cooldown {
                    edge.state
                        .write()
                        .unwrap()
                        .add_cooldown(&Duration::from_secs(30));
                }
                warn!(edge = edge.desc(), amount, "quote is zero, skipping");
                return Ok(None);
            }

            let Some(price) = edge
                .state
                .read()
                .unwrap()
                .cached_price_for(current_in_amount)
            else {
                return Ok(None);
            };

            current_in_amount = quote.out_amount;
            current_in_amount_dumb = ((quote.in_amount as f64) * price.0).round() as u64;

            if current_in_amount_dumb > current_in_amount.saturating_mul(3) {
                if add_cooldown {
                    edge.state
                        .write()
                        .unwrap()
                        .add_cooldown(&Duration::from_secs(30));
                }
                warn!(
                    out_quote = quote.out_amount,
                    out_dumb = current_in_amount_dumb,
                    in_quote = quote.in_amount,
                    price = price.0,
                    edge = edge.desc(),
                    input_mint = debug_tools::name(&edge.input_mint),
                    output_mint = debug_tools::name(&edge.output_mint),
                    prices = edge
                        .state
                        .read()
                        .unwrap()
                        .cached_prices
                        .iter()
                        .map(|x| format!("in={}, price={}", x.0, x.1))
                        .join("||"),
                    "recomputed path amount diverge a lot from estimation - path ignored"
                );
                return Ok(None);
            }
        }
        Ok(Some((current_in_amount, current_in_amount_dumb)))
    }

    fn compute_in_amount_from_path(
        chain_data: &AccountProviderView,
        snap: &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
        path: &[Arc<Edge>],
        amount: u64,
        add_cooldown: bool,
    ) -> anyhow::Result<Option<(u64, u64)>> /* (报价, 缓存价) */ {
        let prepare = Self::prepare;

        let mut current_out_amount = amount;
        let mut current_out_amount_dumb = amount;
        for edge in path {
            if !edge.supports_exact_out() {
                return Ok(None);
            }

            if !edge.state.read().unwrap().is_valid() {
                warn!(edge = edge.desc(), "invalid edge");
                return Ok(None);
            }
            let prepared_quote = match prepare(snap, edge, chain_data) {
                Some(p) => p,
                _ => bail!(RoutingError::CouldNotComputeOut),
            };
            let quote_res = edge.quote_exact_out(&prepared_quote, chain_data, current_out_amount);
            let Ok(quote) = quote_res else {
                if add_cooldown {
                    edge.state
                        .write()
                        .unwrap()
                        .add_cooldown(&Duration::from_secs(30));
                }
                warn!(
                    edge = edge.desc(),
                    amount,
                    "failed to quote, err: {:?}",
                    quote_res.unwrap_err()
                );
                return Ok(None);
            };

            if quote.out_amount == 0 {
                if add_cooldown {
                    edge.state
                        .write()
                        .unwrap()
                        .add_cooldown(&Duration::from_secs(30));
                }
                warn!(edge = edge.desc(), amount, "quote is zero, skipping");
                return Ok(None);
            }

            let Some(price) = edge
                .state
                .read()
                .unwrap()
                .cached_price_exact_out_for(amount)
            else {
                return Ok(None);
            };

            current_out_amount = quote.in_amount;
            current_out_amount_dumb = ((quote.out_amount as f64) * price.0).round() as u64;

            if current_out_amount_dumb > current_out_amount.saturating_mul(3) {
                if add_cooldown {
                    edge.state
                        .write()
                        .unwrap()
                        .add_cooldown(&Duration::from_secs(30));
                }
                warn!(
                    out_quote = quote.out_amount,
                    out_dumb = current_out_amount_dumb,
                    in_quote = quote.in_amount,
                    price = price.0,
                    edge = edge.desc(),
                    input_mint = debug_tools::name(&edge.input_mint),
                    output_mint = debug_tools::name(&edge.output_mint),
                    prices = edge
                        .state
                        .read()
                        .unwrap()
                        .cached_prices
                        .iter()
                        .map(|x| format!("in={}, price={}", x.0, x.1))
                        .join("||"),
                    "recomputed path amount diverge a lot from estimation - path ignored"
                );
                return Ok(None);
            }
        }
        Ok(Some((current_out_amount, current_out_amount_dumb)))
    }

    // 接收输入代币、输出代币、交易金额等参数
    pub fn find_best_route(
        &self,
        chain_data: &AccountProviderView,
        input_mint: &Pubkey,
        output_mint: &Pubkey,
        original_amount: u64,
        max_accounts: usize,
        ignore_cache: bool,
        hot_mints: &HashSet<Pubkey>,
        max_path_length: Option<usize>,
        swap_mode: SwapMode,
    ) -> anyhow::Result<Route> {
        //这是数据预处理。autobahn 会连接很多交易池，但不是所有都有效。这行代码的作用是确保那些无效的、
        // 或者流动性极差的交易池已经被“修剪”掉了，只留下一批有价值的交易池用于后续的路径搜索。这能极大减少计算量。
        self.prepare_pruned_edges_if_not_initialized(hot_mints, swap_mode);

        // 首先尝试使用比最大授权跳数少一跳的路径，因为这样会快很多（20-30%）
        // 如果找不到任何路径，会增加一跳再试一次
        // 设置路径长度 (max_path_length): 这里有一个重要的性能优化。它不会立即使用配置中允许的最大路径长度（比如4跳），
        // 而是先用一个更短的长度（比如3跳）来尝试。因为短路径的搜索空间小得多，计算速度会快很多。如果用短路径找不到，后续才有机会用长路径重试。
        let max_path_length = max_path_length.unwrap_or((self.max_path_length - 1).max(1));

        //高估输入金额 (overquote): 为了防止因为滑点等问题导致最终交易失败，代码会故意将请求的交易金额稍微上浮一点（比如 +20%）。
        //它用这个高估的金额去寻找路径，确保找到的路径有足够的“缓冲”，能成功执行实际金额的交易。
        let amount = (original_amount as f64 * (1.0 + self.overquote)).round() as u64;

        // 签名者 + autobahn-executor程序 + token程序 + 源代币账户（其他的在交易边里）
        // + ATA程序 + 系统程序 + mint
        let min_accounts_needed = 7;

        // 主要步骤:
        // 1. 路径发现：哪些路径可能是比较好的？（这个过程开销大，应该被缓存）
        // 2. 路径评估/优化：进行实际的路径报价，尝试寻找多路径路由
        // 3. 生成输出

        // 在算法内部，直接使用长长的 Pubkey 地址进行计算效率很低。所以项目在初始化时，
        // 会给每个代币地址（Pubkey）分配一个独一无二的、从0开始的数字ID（MintNodeIndex）。
        // 这两行代码就是把用户传进来的 input_mint 和 output_mint 地址，转换成内部使用的数字ID，即 input_index 和 output_index。
        let Some(&input_index) = self.mint_to_index.get(input_mint) else {
            bail!(RoutingError::UnsupportedInputMint(input_mint.clone()));
        };
        let Some(&output_index) = self.mint_to_index.get(output_mint) else {
            bail!(RoutingError::UnsupportedOutputMint(output_mint.clone()));
        };

        trace!(
            input_index = input_index.idx_raw(),
            output_index = output_index.idx_raw(),
            max_path_length,
            "find_best_route"
        );

        // 路径发现：为交易对寻找候选路径。
        // 尽可能优先使用缓存中的路径。
        // 这是函数最核心的部分，目的是找到从起点到终点的所有“候选路径”。
        let cached_paths_opt = {
            //首先尝试从缓存里拿结果。路径搜索非常耗时，所以系统会将每次的搜索结果缓存起来。
            let mut cache = self.path_discovery_cache.write().unwrap();
            //检查缓存里有没有从input_index到output_index，在当前金额amount和账户数max_accounts限制下的记录。
            let cached = cache.get(input_index, output_index, swap_mode, amount, max_accounts);

            //self.lookup_edge_index_paths(...): 如果缓存里有，缓存存的只是路径的ID序列。这两行代码把ID序列转换回完整的路径数据。
            let p1 = cached
                .0
                .map(|paths| self.lookup_edge_index_paths(paths.iter()));
            let p2 = cached
                .1
                .map(|paths| self.lookup_edge_index_paths(paths.iter()));

            //如果调用者强制要求ignore_cache，或者缓存里确实什么都没有，那么cached_paths_opt就会是 None。否则，它会包含所有从缓存中读出的路径。
            if (p1.is_none() && p2.is_none()) || ignore_cache {
                None
            } else {
                let cached_paths = p1
                    .unwrap_or(vec![])
                    .into_iter()
                    .chain(p2.unwrap_or(vec![]).into_iter())
                    .collect_vec();
                Some(cached_paths)
            }
        };

        let timestamp = millis_since_epoch();
        let pruned = match swap_mode {
            SwapMode::ExactIn => self
                .pruned_out_edges_per_mint_index_exact_in
                .read()
                .unwrap(),
            SwapMode::ExactOut => self
                .pruned_out_edges_per_mint_index_exact_out
                .read()
                .unwrap(),
        };
        let out_edges_per_node = &pruned.1;

        let mut paths;
        let mut used_cached_paths = false;
        // 如果缓存里有路径 (if let Some(...))，就直接用缓存的路径。如果缓存里没有 (else)，就启动一次全新的、实时的搜索。

        if let Some(cached_paths) = cached_paths_opt {
            paths = cached_paths;
            used_cached_paths = true;
        } else {
            let avoid_cold_mints = !ignore_cache;
            let hot_mints = hot_mints
                .iter()
                .filter_map(|x| self.mint_to_index.get(x))
                .copied()
                .collect();

            let (out_paths, new_paths_by_out_node) = match swap_mode {
                SwapMode::ExactIn => {
                    // 这是真正启动搜索的地方。它会调用底层的深度优先搜索算法，在所有（经过修剪的）交易池网络里，
                    // 从input_index开始，查找所有能在max_path_length步内到达其他所有节点的路径
                    let new_paths_by_out_node = self.generate_best_paths(
                        amount,
                        timestamp,
                        max_accounts,
                        min_accounts_needed,
                        input_index,
                        &out_edges_per_node,
                        &hot_mints,
                        avoid_cold_mints,
                        max_path_length,
                    )?;
                    (
                        // generate_best_paths 返回的是从起点到所有节点的路径。这一小段代码是从这批海量的结果中，
                        // 只把那些终点恰好是我们想要的output_index的路径筛选出来，存入out_paths
                        self.lookup_edge_index_paths(new_paths_by_out_node[output_index].iter()),
                        new_paths_by_out_node,
                    )
                }
                SwapMode::ExactOut => {
                    let new_paths_by_out_node = self.generate_best_paths_exact_out(
                        amount,
                        timestamp,
                        max_accounts,
                        min_accounts_needed,
                        output_index,
                        &out_edges_per_node,
                        &hot_mints,
                        avoid_cold_mints,
                        max_path_length,
                    )?;
                    (
                        self.lookup_edge_index_paths(new_paths_by_out_node[input_index].iter()),
                        new_paths_by_out_node,
                    )
                }
            };

            //路径评估 - 计算每条路径的真实价格 (1365-1395行)

            //现在，变量paths里装着所有候选路径。但这些路径在搜索时用的是预估价格，不一定准。这一步就是要对它们进行精确的计算和排序。
            paths = out_paths;

            for (out_index, new_paths) in new_paths_by_out_node.into_iter().enumerate() {
                let out_index: MintNodeIndex = out_index.into();
                // self.add_direct_paths(...): 这是一个保险措施。它会把所有从起点到终点的“单跳”路径
                //（即只经过一个交易池的路径）也手动加到paths列表里，确保最简单的路径不会被遗漏
                self.path_discovery_cache.write().unwrap().insert(
                    input_index,
                    out_index,
                    swap_mode,
                    amount,
                    max_accounts,
                    millis_since_epoch(),
                    new_paths,
                );
            }
        }

        // 路径发现：添加所有直接路径
        // 注意：目前这可能意味着某些路径会存在两次
        self.add_direct_paths(input_index, output_index, out_edges_per_node, &mut paths);

        // 不要保持锁定状态——这会导致递归死锁，并且会影响性能
        drop(pruned);

        // 路径评估
        // TODO: 这里可以评估路径对，看是否进行拆分交易更有意义，
        // 不过需要处理好共享步骤的问题。

        let mut snapshot = HashMap::new();

        // 这个函数（compute_out_amount_from_path 或 compute_in_amount_from_path）会模拟这笔交易，
        // 考虑所有真实的手续费、滑点等因素，计算出这条路径最终能得到的精确兑换数量
        let path_output_fn = match swap_mode {
            SwapMode::ExactIn => Self::compute_out_amount_from_path,
            SwapMode::ExactOut => Self::compute_in_amount_from_path,
        };

        //最终，path_and_output变量里会存储一个列表，每个元素是 (路径, 真实兑换数量, 预估兑换数量)
        let mut path_and_output = paths
            .into_iter()
            .filter_map(|path| {
                path_output_fn(chain_data, &mut snapshot, &path, amount, true)
                    .ok()
                    .flatten()
                    .filter(|v| v.0 > 0)
                    .map(|v| (path, v.0, v.1))
            })
            .collect_vec();

        //根据上一步计算出的真实兑换数量，对所有路径进行排序。
        // 买入时，按能收到的out_amount从高到低排；
        // 卖出时，按需要付出的in_amount从低到高排。这样，列表的第一条路径，就是理论上的“最佳路径”。
        match swap_mode {
            SwapMode::ExactIn => path_and_output.sort_by_key(|(_, v, _)| std::cmp::Reverse(*v)),
            SwapMode::ExactOut => path_and_output.sort_by_key(|(_, v, _)| *v),
        }

        // 调试
        if tracing::event_enabled!(Level::TRACE) {
            for (path, out_amount, out_amount_dumb) in &path_and_output {
                trace!(
                    "potential path: [out={}] [dumb={}] {}",
                    out_amount,
                    out_amount_dumb,
                    path.iter().map(|edge| edge.desc()).join(", ")
                );
            }
        }

        //第4部分：构建最终路由并返回 (1411-1517行)
        // 现在我们有了一个按优劣排好序的路径列表。这一步就是从最好的那条开始，尝试构建一个可以被执行的、完整的Route对象。

        //从最好的一条路径开始，循环尝试。
        for (out_path, routing_result, _) in path_and_output {
            let (route_steps, context_slot) = match swap_mode {
                // 在这里恢复请求的 `in_amount` 用于构建路由
                //作用: 将抽象的路径信息（比如经过哪几个交易池），转换成具体的交易步骤Vec<RouteStep>。
                //注意，这里用的是用户原始请求的original_amount，而不是我们之前为了安全垫而高估的amount。
                SwapMode::ExactIn => Self::build_route_steps(
                    chain_data,
                    &mut snapshot,
                    Self::prepare,
                    &out_path,
                    original_amount,
                )?,
                SwapMode::ExactOut => Self::build_route_steps_exact_out(
                    chain_data,
                    &mut snapshot,
                    Self::prepare,
                    &out_path,
                    original_amount,
                )?,
            };

            // 从构建好的route_steps中，提取出经过精确计算后的实际输入和输出数量。
            let actual_in_amount = route_steps.first().unwrap().in_amount;
            let actual_out_amount = route_steps.last().unwrap().out_amount;

            let overquote_in_amount = match swap_mode {
                SwapMode::ExactIn => amount,
                SwapMode::ExactOut => routing_result,
            };

            // 获取一个“几乎没有价格影响”的基准汇率。
            // 它调用Self::compute_out_amount_from_path函数，模拟用同样的路径，但只交易一笔极小的金额（1_000个最小单位）。
            // 因为金额小，所以它几乎不会影响市场价格，得到的汇率可以看作是当前的“即时价”或“公允价”。
            let out_amount_for_small_amount = Self::compute_out_amount_from_path(
                chain_data,
                &mut snapshot,
                &out_path,
                1_000,
                false,
            )?
            .unwrap_or_default()
            .0;

            //let out_amount_for_request = actual_out_amount;
            //目的: 这只是给变量起一个更清晰的名字，actual_out_amount是您这笔真实金额交易算出来的最终输出数量。
            let out_amount_for_request = actual_out_amount;
            //计算并比较两种情况下的汇率
            //expected_ratio: 小额交易的“理想”汇率。
            //actual_ratio: 您这笔大额交易的“实际”汇率。
            let expected_ratio = 1_000.0 / out_amount_for_small_amount as f64;
            let actual_ratio = actual_in_amount as f64 / actual_out_amount as f64;

            // 通过比较“理想”汇率和“实际”汇率的差异，计算出一个百分比，通常用基点（BPS）表示（100 BPS = 1%）。
            // 这个price_impact_bps值会包含在最终返回的路由信息里，让用户知道这笔交易的滑点有多大
            let price_impact = expected_ratio / actual_ratio * 10_000.0 - 10_000.0;
            let price_impact_bps = price_impact.round() as u64;

            trace!(
                price_impact_bps,
                out_amount_for_small_amount,
                out_amount_for_request,
                expected_ratio,
                actual_ratio,
                "price impact"
            );

            let adjusted_out_amount = match swap_mode {
                //这段公式 ((actual_in_amount / overquote_in_amount) * routing_result) 看起来复杂，其实是在做线性缩放。
                //它的逻辑是：“既然我的实际输入金额(actual_in_amount)是高估输入金额(overquote_in_amount)的X倍，那
                //么我最终得到的输出金额，也应该是高估金额算出来的输出(routing_result)的X倍”。这是一种近似计算，
                //用于将输出金额调整回与原始输入金额相匹配的水平
                SwapMode::ExactIn => ((actual_in_amount as f64 / overquote_in_amount as f64)
                    * routing_result as f64)
                    .floor() as u64,
                SwapMode::ExactOut => original_amount,
            };

            let adjusted_in_amount = match swap_mode {
                SwapMode::ExactIn => actual_in_amount,
                SwapMode::ExactOut => ((actual_out_amount as f64 / amount as f64)
                    * routing_result as f64)
                    .ceil() as u64,
            };

            let out_amount_for_overquoted_amount = match swap_mode {
                SwapMode::ExactIn => routing_result,
                SwapMode::ExactOut => amount,
            };

            //这是一个守卫语句，防止返回无效或无意义的路径
            //它检查：对于“精确输出”模式，计算出的所需输入金额是不是无穷大（u64::MAX）？对于“精确输入”模式，计算出的最终输出金额是不是0？
            //如果出现这些情况，说明这条路径实际上是走不通的。continue;语句会跳过这条路径，继续尝试候选列表中的下一条。
            if (swap_mode == SwapMode::ExactOut
                && (actual_in_amount == u64::MAX || actual_out_amount == 0))
                || (swap_mode == SwapMode::ExactIn && adjusted_out_amount == 0)
            {
                continue;
            }

            if self.overquote > 0.0 {
                debug!(
                    actual_in_amount,
                    actual_out_amount,
                    overquote_in_amount,
                    out_amount_for_overquoted_amount,
                    adjusted_out_amount,
                    "adjusted amount"
                );
            }

            // If enabled,for debug purpose, recompute route while capturing accessed chain accounts
            // Can be used when executing the swap to check if accounts have changed
            // 目的: 这是一个可选的调试和安全功能。如果启用了某个编译特性（capture-accounts），它会再次模拟交易，并把这条路径上所有涉及到的账户的当前状态（数据和版本号）“快照”并记录下来。
            // 用途: 在真正执行这笔交易前，可以再次检查这些账户的状态是否和快照时一致。如果有变化，可能意味着价格已经变了，此时可以中止交易，防止用户遭受损失。
            let accounts = self
                .capture_accounts(chain_data, &out_path, original_amount)
                .ok();

            //成功退出点。一旦一条路径通过了以上所有的检查和计算，代码就会在这里创建一个最终的 Route 对象，
            // 打包所有信息（输入输出金额、交易步骤、价格影响、账户快照等），然后通过 return Ok(...) 成功返回。
            // 整个find_best_route函数在此刻就结束了。
            return Ok(Route {
                input_mint: *input_mint,
                output_mint: *output_mint,
                in_amount: adjusted_in_amount,
                out_amount: adjusted_out_amount,
                steps: route_steps,
                slot: context_slot,
                price_impact_bps,
                accounts,
            });
        }

        //处理完全失败的情况
        //即上面的for循环把所有候选路径都试了一遍，但没有一条能够通过检查并成功返回。
        // 目的: 启动“B计划”，进行最后一次尝试。
        // 如何做: 它会调用自己，但这次会放宽限制，例如：
        // 如果之前用了缓存，这次就强制忽略缓存，因为缓存可能已经过时了。
        // 如果之前没用缓存，说明在当前路径长度限制下找不到，这次就将最大路径长度加一，扩大搜索范围。
        // 如果这“最后的尝试”成功了，它会返回一个Route。如果连这次都失败了，那程序就真的回天乏术，最终会向上层抛出 NoPathBetweenMintPair 错误。
        // No acceptable path
        self.find_route_with_relaxed_constraints_or_fail(
            &chain_data,
            input_mint,
            swap_mode,
            output_mint,
            amount,
            max_accounts,
            ignore_cache,
            hot_mints,
            max_path_length,
            input_index,
            output_index,
            used_cached_paths,
        )
    }

    fn find_route_with_relaxed_constraints_or_fail(
        &self,
        chain_data: &AccountProviderView,
        input_mint: &Pubkey,
        swap_mode: SwapMode,
        output_mint: &Pubkey,
        amount: u64,
        max_accounts: usize,
        ignore_cache: bool,
        hot_mints: &HashSet<Pubkey>,
        max_path_length: usize,
        input_index: MintNodeIndex,
        output_index: MintNodeIndex,
        used_cached_paths: bool,
    ) -> anyhow::Result<Route> {
        // It is possible for cache path to became invalid after some account write or failed tx (cooldown)
        // If we used cache but can't find any valid path, try again without the cache
        let can_try_one_more_hop = max_path_length != self.max_path_length;
        if !ignore_cache && (used_cached_paths || can_try_one_more_hop) {
            if used_cached_paths {
                debug!("Invalid cached path, retrying without cache");
                let mut cache = self.path_discovery_cache.write().unwrap();
                cache.invalidate(input_index, output_index, max_accounts);
            } else {
                debug!("No path within boundaries, retrying with +1 hop");
            }
            return self.find_best_route(
                chain_data,
                input_mint,
                output_mint,
                amount,
                max_accounts,
                used_cached_paths,
                hot_mints,
                Some(self.max_path_length),
                swap_mode,
            );
        }

        // self.print_debug_data(input_mint, output_mint, max_accounts);

        bail!(RoutingError::NoPathBetweenMintPair(
            input_mint.clone(),
            output_mint.clone()
        ));
    }

    fn capture_accounts(
        &self,
        chain_data: &AccountProviderView,
        out_path: &Vec<Arc<Edge>>,
        original_in_amount: u64,
    ) -> anyhow::Result<HashMap<Pubkey, AccountData>> {
        #[cfg(not(feature = "capture-accounts"))]
        return Ok(Default::default());

        #[cfg(feature = "capture-accounts")]
        {
            let mut snapshot = HashMap::new();
            let prepare = |s: &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
                           e: &Arc<Edge>,
                           c: &AccountProviderView|
             -> Option<Arc<dyn DexEdge>> {
                s.entry(e.unique_id())
                    .or_insert_with(move || e.prepare(c).ok())
                    .clone()
            };

            let chain_data = Arc::new(ChainDataCaptureAccountProvider::new(chain_data.clone()));
            let downcasted = chain_data.clone() as AccountProviderView;

            Self::build_route_steps(
                &downcasted,
                &mut snapshot,
                prepare,
                &out_path,
                original_in_amount,
            )?;

            let accounts = chain_data.accounts.read().unwrap().clone();
            for (acc, _) in &accounts {
                if !debug_tools::is_in_global_filters(acc) {
                    error!(
                        "Used an account that we are not listening to, addr={:?} !",
                        acc
                    )
                }
            }

            Ok(accounts)
        }
    }

    // 每个请求一次，然后是路径中的每个边
    #[tracing::instrument(skip_all, level = "trace")]
    fn build_route_steps(
        chain_data: &AccountProviderView,
        mut snapshot: &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
        prepare: fn(
            &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
            &Arc<Edge>,
            &AccountProviderView,
        ) -> Option<Arc<dyn DexEdge>>,
        out_path: &Vec<Arc<Edge>>,
        in_amount: u64,
    ) -> anyhow::Result<(Vec<RouteStep>, u64)> {
        let mut context_slot = 0;
        let mut steps = Vec::with_capacity(out_path.len());
        let mut current_in_amount = in_amount;
        for edge in out_path.iter() {
            let prepared_quote = match prepare(&mut snapshot, edge, chain_data) {
                Some(p) => p,
                _ => bail!(RoutingError::CouldNotComputeOut),
            };

            let quote = edge.quote(&prepared_quote, chain_data, current_in_amount)?;
            steps.push(RouteStep {
                edge: edge.clone(),
                in_amount: quote.in_amount,
                out_amount: quote.out_amount,
                fee_amount: quote.fee_amount,
                fee_mint: quote.fee_mint,
            });
            current_in_amount = quote.out_amount;
            let edge_slot = edge.state.read().unwrap().last_update_slot;
            context_slot = edge_slot.max(context_slot);
        }

        Ok((steps, context_slot))
    }

    #[tracing::instrument(skip_all, level = "trace")]
    fn build_route_steps_exact_out(
        chain_data: &AccountProviderView,
        mut snapshot: &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
        prepare: fn(
            &mut HashMap<(Pubkey, Pubkey), Option<Arc<dyn DexEdge>>>,
            &Arc<Edge>,
            &AccountProviderView,
        ) -> Option<Arc<dyn DexEdge>>,
        out_path: &Vec<Arc<Edge>>,
        out_amount: u64,
    ) -> anyhow::Result<(Vec<RouteStep>, u64)> {
        let mut context_slot = 0;
        let mut steps = Vec::with_capacity(out_path.len());
        let mut current_out_amount = out_amount;
        for edge in out_path.iter() {
            let prepared_quote = match prepare(&mut snapshot, edge, chain_data) {
                Some(p) => p,
                _ => bail!(RoutingError::CouldNotComputeOut),
            };

            let quote = edge.quote_exact_out(&prepared_quote, chain_data, current_out_amount)?;
            steps.push(RouteStep {
                edge: edge.clone(),
                in_amount: quote.in_amount,
                out_amount: quote.out_amount,
                fee_amount: quote.fee_amount,
                fee_mint: quote.fee_mint,
            });
            current_out_amount = quote.in_amount;
            let edge_slot = edge.state.read().unwrap().last_update_slot;
            context_slot = edge_slot.max(context_slot);
        }

        // 对于 exact out，反转步骤
        steps.reverse();

        Ok((steps, context_slot))
    }

    fn add_direct_paths(
        &self,
        input_index: MintNodeIndex,
        output_index: MintNodeIndex,
        out_edges_per_node: &MintVec<Vec<EdgeWithNodes>>,
        paths: &mut Vec<Vec<Arc<Edge>>>,
    ) {
        for target_and_edge in &out_edges_per_node[input_index] {
            if target_and_edge.target_node != output_index {
                continue;
            }
            let edge = &self.edges[target_and_edge.edge.idx()];
            let state = edge.state.read().unwrap();
            if !state.is_valid() {
                continue;
            }

            if !paths
                .iter()
                .any(|x| x.len() == 1 && x[0].unique_id() == edge.unique_id())
            {
                paths.push(vec![edge.clone()]);
            }
        }
    }

    // 每个请求调用一次
    #[tracing::instrument(skip_all, level = "trace")]
    fn generate_best_paths(
        &self,
        in_amount: u64,
        now_ms: u64,
        max_accounts: usize,
        min_accounts_needed: usize,
        input_index: MintNodeIndex,
        out_edges_per_node: &MintVec<Vec<EdgeWithNodes>>,
        hot_mints: &HashSet<MintNodeIndex>,
        avoid_cold_mints: bool,
        max_path_length: usize,
    ) -> anyhow::Result<MintVec<Vec<Vec<EdgeIndex>>>> {
        // 非池化版本
        // let mut best_by_node_prealloc = vec![vec![0f64; 3]; 8 * out_edges_per_node.len()];
        let mut best_by_node_prealloc = self.objectpools.get_best_by_node(out_edges_per_node.len());

        // 非池化版本
        // let mut best_paths_by_node_prealloc: MintVec<Vec<(NotNan<f64>, Vec<EdgeWithNodes>)>> =
        //     MintVec::new_from_prototype(
        //         out_edges_per_node.len(),
        //         vec![(NotNan::new(f64::NEG_INFINITY).unwrap(), vec![]); self.retain_path_count],
        //     );
        let mut best_paths_by_node_prealloc = self
            .objectpools
            .get_best_paths_by_node(out_edges_per_node.len(), self.retain_path_count);

        let new_paths_by_out_node = best_price_paths_depth_search(
            input_index,
            in_amount,
            max_path_length,
            max_accounts.saturating_sub(min_accounts_needed),
            out_edges_per_node,
            &mut best_paths_by_node_prealloc,
            &mut best_by_node_prealloc,
            |edge_index, in_amount| self.edge_info(edge_index, now_ms, in_amount),
            hot_mints,
            avoid_cold_mints,
            SwapMode::ExactIn,
        )?;

        let mut best_paths_by_out_node =
            MintVec::new_from_prototype(new_paths_by_out_node.len(), vec![]);
        // 长度大约是 400000
        trace!("new_paths_by_out_node len={}", new_paths_by_out_node.len());

        for (out_node, best_paths) in new_paths_by_out_node.into_iter().enumerate() {
            let out_node: MintNodeIndex = out_node.into();
            let mut paths = Vec::with_capacity(best_paths.len());

            for (_, path) in best_paths {
                let edges = path.into_iter().map(|edge| edge.edge).collect();
                paths.push(edges);
            }

            best_paths_by_out_node[out_node] = paths;
        }

        Ok(best_paths_by_out_node)
    }

    // 每个请求调用一次
    #[tracing::instrument(skip_all, level = "trace")]
    fn generate_best_paths_exact_out(
        &self,
        out_amount: u64,
        now_ms: u64,
        max_accounts: usize,
        min_accounts_needed: usize,
        output_index: MintNodeIndex,
        out_edges_per_node: &MintVec<Vec<EdgeWithNodes>>,
        hot_mints: &HashSet<MintNodeIndex>,
        avoid_cold_mints: bool,
        max_path_length: usize,
    ) -> anyhow::Result<MintVec<Vec<Vec<EdgeIndex>>>> {
        // 类似于 generate_best_paths，只是改变了计算边信息的函数并将 is_exact_out 设置为 true
        let mut best_by_node_prealloc = self.objectpools.get_best_by_node(out_edges_per_node.len());

        let mut best_paths_by_node_prealloc = self
            .objectpools
            .get_best_paths_by_node_exact_out(out_edges_per_node.len(), self.retain_path_count);

        let new_paths_by_out_node = best_price_paths_depth_search(
            output_index,
            out_amount,
            max_path_length,
            max_accounts.saturating_sub(min_accounts_needed),
            out_edges_per_node,
            &mut best_paths_by_node_prealloc,
            &mut best_by_node_prealloc,
            |edge_index, out_amount: u64| self.edge_info_exact_out(edge_index, now_ms, out_amount),
            hot_mints,
            avoid_cold_mints,
            SwapMode::ExactOut,
        )?;

        let mut best_paths_by_out_node =
            MintVec::new_from_prototype(new_paths_by_out_node.len(), vec![]);
        trace!("new_paths_by_out_node len={}", new_paths_by_out_node.len());

        for (out_node, best_paths) in new_paths_by_out_node.into_iter().enumerate() {
            let out_node: MintNodeIndex = out_node.into();
            let mut paths = Vec::with_capacity(best_paths.len());

            for (_, path) in best_paths {
                let edges = path.into_iter().map(|edge| edge.edge).collect();
                paths.push(edges);
            }

            best_paths_by_out_node[out_node] = paths;
        }

        Ok(best_paths_by_out_node)
    }

    pub fn find_edge(
        &self,
        input_mint: Pubkey,
        output_mint: Pubkey,
        amm_key: Pubkey,
    ) -> anyhow::Result<Arc<Edge>> {
        if let Some(result) = self.edges.iter().find(|x| {
            x.input_mint == input_mint && x.output_mint == output_mint && x.key() == amm_key
        }) {
            Ok(result.clone())
        } else {
            Err(anyhow::format_err!("Edge not found"))
        }
    }

    fn print_debug_data(&self, input_mint: &Pubkey, output_mint: &Pubkey, max_accounts: usize) {
        warn!(
            %input_mint,
            %output_mint, max_accounts, "Couldn't find a path"
        );

        let mut seen_out = HashSet::new();
        let mut seen_in = HashSet::new();
        for edge in &self.edges {
            if edge.output_mint == *output_mint {
                Self::print_some_edges(&mut seen_out, edge);
            }
            if edge.input_mint == *input_mint {
                Self::print_some_edges(&mut seen_in, edge);
            }
        }
    }

    fn print_some_edges(seen: &mut HashSet<(Pubkey, Pubkey)>, edge: &Arc<Edge>) {
        if seen.insert(edge.unique_id()) == false || seen.len() > 6 {
            return;
        }
        let (valid, prices) = {
            let reader = edge.state.read().unwrap();
            let prices = reader
                .cached_prices
                .iter()
                .map(|x| format!("q={} @ p={}", x.0, x.1))
                .join(" // ");
            (reader.is_valid(), prices)
        };
        warn!(
            edge = edge.id.desc(),
            input = debug_tools::name(&edge.input_mint),
            output = debug_tools::name(&edge.output_mint),
            valid,
            prices,
            " - available edge"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::test::MockDexIdentifier;
    use crate::mock::test::MockDexInterface;
    use crate::routing_objectpool::{
        alloc_best_by_node_for_test, alloc_best_paths_by_node_for_test,
    };
    use router_lib::chain_data::ChainDataArcRw;
    use router_lib::dex::{ChainDataAccountProvider, DexInterface};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    fn make_graph(
        edges: &[(&str, &str, f64)],
    ) -> (
        HashMap<String, MintNodeIndex>,
        MintVec<Vec<EdgeWithNodes>>,
        Vec<f64>,
    ) {
        let edge_prices = edges.iter().map(|(_, _, p)| *p).collect_vec();

        let mut nodes = HashMap::<String, MintNodeIndex>::new();
        let mut add_node = |name: &str| {
            if !nodes.contains_key(name) {
                nodes.insert(name.to_string(), nodes.len().into());
            }
        };
        for (from, to, _) in edges {
            add_node(from);
            add_node(to);
        }

        let mut out_edges_per_node = MintVec::new_from_prototype(nodes.len(), vec![]);
        for (edge_index, (from, to, _)) in edges.iter().enumerate() {
            let edge_index = edge_index.into();
            let from_index = nodes.get(&from.to_string()).unwrap();
            let to_index = nodes.get(&to.to_string()).unwrap();
            out_edges_per_node[*from_index].push(EdgeWithNodes {
                edge: edge_index,
                source_node: *from_index,
                target_node: *to_index,
            });
        }

        (nodes, out_edges_per_node, edge_prices)
    }

    macro_rules! assert_eq_f64 {
        ($actual:expr, $expected:expr, $delta:expr) => {
            if ($actual - $expected).abs() > $delta {
                println!(
                    "assertion failed: {} is not approximately {}",
                    $actual, $expected
                );
                assert!(false);
            }
        };
    }

    macro_rules! check_path {
        ($path:expr, $expected_cost:expr, $expected_nodes:expr, $node_lookup:expr) => {
            let path = $path;
            let node_lookup = $node_lookup;
            let path_nodes = path.1.iter().map(|edge| edge.target_node).collect_vec();
            let expected_path_nodes = $expected_nodes
                .iter()
                .map(|node| *node_lookup.get(&node.to_string()).unwrap())
                .collect_vec();
            assert_eq!(path_nodes, expected_path_nodes);
            assert_eq_f64!(path.0, $expected_cost, 0.000001);
        };
    }

    #[test]
    fn find_best_paths() {
        let (nodes, out_edges_per_node, edge_prices) = make_graph(&[
            ("USDC", "USDT", 0.99),
            ("USDC", "USDT", 0.95),
            ("USDT", "USDC", 2.0), // never taken when starting from USDC: no cycles allowed
            ("USDC", "SOL", 0.99 / 200.0),
            ("SOL", "USDT", 0.98 * 200.0),
            ("SOL", "DAI", 1.05 * 200.0),
            ("DAI", "USDT", 0.97),
        ]);
        let edge_info_fn = |edge: EdgeIndex, _in_amount| {
            Some(EdgeInfo {
                price: edge_prices[edge.idx()],
                accounts: 0,
            })
        };
        let get_paths = |from: &str, to: &str, max_length| {
            best_price_paths_depth_search(
                *nodes.get(&from.to_string()).unwrap(),
                1,
                max_length,
                64,
                &out_edges_per_node,
                &mut alloc_best_paths_by_node_for_test(
                    out_edges_per_node.len(),
                    5,
                    SwapMode::ExactIn,
                ),
                &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                edge_info_fn,
                &HashSet::new(),
                false,
                SwapMode::ExactIn,
            )
            .unwrap()[*nodes.get(&to.to_string()).unwrap()]
            .clone()
        };

        {
            let paths = get_paths("USDC", "USDT", 1);
            check_path!(&paths[0], 0.99, &["USDT"], &nodes);
            check_path!(&paths[1], 0.95, &["USDT"], &nodes);
            assert_eq!(paths.len(), 2);
        }
        {
            let paths = get_paths("USDC", "USDT", 2);
            check_path!(&paths[0], 0.99, &["USDT"], &nodes);
            check_path!(&paths[1], 0.99 * 0.98, &["SOL", "USDT"], &nodes);
            check_path!(&paths[2], 0.95, &["USDT"], &nodes);
            assert_eq!(paths.len(), 3);
        }
        {
            let paths = get_paths("USDC", "USDT", 3);
            check_path!(
                &paths[0],
                0.99 * 1.05 * 0.97,
                &["SOL", "DAI", "USDT"],
                &nodes
            );
            check_path!(&paths[1], 0.99, &["USDT"], &nodes);
            check_path!(&paths[2], 0.99 * 0.98, &["SOL", "USDT"], &nodes);
            assert_eq!(paths.len(), 4);
        }
    }

    #[test]
    fn find_best_paths_exact_out() {
        let (nodes, out_edges_per_node, edge_prices) = make_graph(&[
            ("USDC", "USDT", 0.99),
            ("USDC", "USDT", 0.95),
            ("USDT", "USDC", 1.05), // never taken when starting from USDC: no cycles allowed
            ("USDC", "SOL", 200.0 * 0.98),
            ("SOL", "USDT", 1.0 / 200.0),
            ("SOL", "DAI", 1.0 / 200.0),
            ("DAI", "USDT", 0.95),
        ]);
        let edge_info_fn = |edge: EdgeIndex, _in_amount| {
            Some(EdgeInfo {
                price: edge_prices[edge.idx()],
                accounts: 0,
            })
        };
        let get_paths = |from: &str, to: &str, max_length| {
            best_price_paths_depth_search(
                *nodes.get(&from.to_string()).unwrap(),
                1,
                max_length,
                64,
                &out_edges_per_node,
                &mut alloc_best_paths_by_node_for_test(
                    out_edges_per_node.len(),
                    5,
                    SwapMode::ExactOut,
                ),
                &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                edge_info_fn,
                &HashSet::new(),
                false,
                SwapMode::ExactOut,
            )
            .unwrap()[*nodes.get(&to.to_string()).unwrap()]
            .clone()
        };

        let paths = get_paths("USDC", "USDT", 3);
        for (price, edges) in paths {
            println!(
                "{} = {}",
                price,
                edges
                    .iter()
                    .map(|x| format!("({:?}: {:?}->{:?})", x.edge, x.source_node, x.target_node))
                    .join(", ")
            )
        }

        {
            let paths = get_paths("USDC", "USDT", 1);
            check_path!(&paths[0], 0.95, &["USDT"], &nodes);
            check_path!(&paths[1], 0.99, &["USDT"], &nodes);
            assert_eq!(paths.len(), 2);
        }
        {
            let paths = get_paths("USDC", "USDT", 2);
            check_path!(&paths[0], 0.95, &["USDT"], &nodes);
            check_path!(&paths[1], 0.98, &["SOL", "USDT"], &nodes);
            check_path!(&paths[2], 0.99, &["USDT"], &nodes);
            assert_eq!(paths.len(), 3);
        }
        {
            let paths = get_paths("USDC", "USDT", 3);
            check_path!(
                &paths[0],
                0.98 * 1.0 * 0.95,
                &["SOL", "DAI", "USDT"],
                &nodes
            );
            check_path!(&paths[1], 0.95, &["USDT"], &nodes);
            check_path!(&paths[2], 0.98, &["SOL", "USDT"], &nodes);
            check_path!(&paths[3], 0.99, &["USDT"], &nodes);
            assert_eq!(paths.len(), 4);
        }
    }

    /// Check that correct paths are found when the edge prices depend on the input amount
    #[test]
    fn find_best_paths_variable_prices() {
        let (nodes, out_edges_per_node, edge_prices) = make_graph(&[
            ("USDC", "USDT", 0.99),        // * 0.5
            ("USDC", "USDT", 0.95),        // * 0.8
            ("USDC", "SOL", 0.99 / 200.0), // * 0.8
            ("SOL", "USDT", 0.98 * 200.0), // * 0.8
            ("USDC", "DAI", 1.01),         // * 0.8
            ("DAI", "USDT", 1.02),         // * 0.8
        ]);
        let edge_price_fn = |edge: EdgeIndex, in_amount| {
            let price = if in_amount > 100 {
                edge_prices[edge.idx()] * if edge.idx() == 0 { 0.5 } else { 0.8 }
            } else {
                edge_prices[edge.idx()]
            };
            Some(EdgeInfo { price, accounts: 0 })
        };
        let get_paths = |from: &str, to: &str, in_amount, max_length| {
            best_price_paths_depth_search(
                *nodes.get(&from.to_string()).unwrap(),
                in_amount,
                max_length,
                64,
                &out_edges_per_node,
                &mut alloc_best_paths_by_node_for_test(
                    out_edges_per_node.len(),
                    5,
                    SwapMode::ExactIn,
                ),
                &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                edge_price_fn,
                &HashSet::new(),
                false,
                SwapMode::ExactIn,
            )
            .unwrap()[*nodes.get(&to.to_string()).unwrap()]
            .clone()
        };

        {
            let paths = get_paths("USDC", "USDT", 1, 1);
            check_path!(&paths[0], 0.99, &["USDT"], &nodes);
            check_path!(&paths[1], 0.95, &["USDT"], &nodes);
            assert_eq!(paths.len(), 2);
        }
        {
            let paths = get_paths("USDC", "USDT", 1000, 1);
            check_path!(&paths[0], 1000.0 * 0.95 * 0.8, &["USDT"], &nodes);
            check_path!(&paths[1], 1000.0 * 0.99 * 0.5, &["USDT"], &nodes);
            assert_eq!(paths.len(), 2);
        }
        {
            let paths = get_paths("USDC", "USDT", 1, 2);
            check_path!(&paths[0], 1.01 * 1.02, &["DAI", "USDT"], &nodes);
            check_path!(&paths[1], 0.99, &["USDT"], &nodes);
            check_path!(&paths[2], 0.99 * 0.98, &["SOL", "USDT"], &nodes);
            check_path!(&paths[3], 0.95, &["USDT"], &nodes);
            assert_eq!(paths.len(), 4);
        }
        {
            let paths = get_paths("USDC", "USDT", 1000, 2);
            check_path!(
                &paths[0],
                1000.0 * 0.99 * 0.98 * 0.8,
                &["SOL", "USDT"],
                &nodes
            );
            check_path!(&paths[1], 1000.0 * 0.95 * 0.8, &["USDT"], &nodes);
            check_path!(
                &paths[2],
                1000.0 * 1.01 * 1.02 * 0.8 * 0.8,
                &["DAI", "USDT"],
                &nodes
            );
            check_path!(&paths[3], 1000.0 * 0.99 * 0.5, &["USDT"], &nodes);
            assert_eq!(paths.len(), 4);
        }
    }

    #[test]
    fn find_best_paths_variable_prices_exact_out() {
        let (nodes, out_edges_per_node, edge_prices) = make_graph(&[
            ("USDC", "USDT", 1.0 / 0.99),          // * 0.5
            ("USDC", "USDT", 1.0 / 0.95),          // * 0.8
            ("USDC", "SOL", 200.0 / 0.99),         // * 0.8
            ("SOL", "USDT", 1.0 / (200.0 * 0.98)), // * 0.8
            ("USDC", "DAI", 1.0 / 1.01),           // * 0.8
            ("DAI", "USDT", 1.0 / 1.02),           // * 0.8
        ]);
        let edge_price_fn = |edge: EdgeIndex, in_amount| {
            let price = if in_amount > 500 {
                edge_prices[edge.idx()] * if edge.idx() == 0 { 0.5 } else { 0.8 }
            } else {
                edge_prices[edge.idx()]
            };
            Some(EdgeInfo { price, accounts: 0 })
        };
        let get_paths = |from: &str, to: &str, in_amount, max_length| {
            best_price_paths_depth_search(
                *nodes.get(&from.to_string()).unwrap(),
                in_amount,
                max_length,
                64,
                &out_edges_per_node,
                &mut alloc_best_paths_by_node_for_test(
                    out_edges_per_node.len(),
                    5,
                    SwapMode::ExactOut,
                ),
                &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                edge_price_fn,
                &HashSet::new(),
                false,
                SwapMode::ExactOut,
            )
            .unwrap()[*nodes.get(&to.to_string()).unwrap()]
            .clone()
        };

        {
            let paths = get_paths("USDC", "USDT", 1, 1);
            check_path!(&paths[0], 1.0 / 0.99, &["USDT"], &nodes);
            check_path!(&paths[1], 1.0 / 0.95, &["USDT"], &nodes);
            assert_eq!(paths.len(), 2);
        }
        {
            let paths = get_paths("USDC", "USDT", 1000, 1);
            check_path!(&paths[0], 1000.0 * 1.0 / 0.99 * 0.5, &["USDT"], &nodes);
            check_path!(&paths[1], 1000.0 * 1.0 / 0.95 * 0.8, &["USDT"], &nodes);
            assert_eq!(paths.len(), 2);
        }
        {
            let paths = get_paths("USDC", "USDT", 1, 2);
            check_path!(&paths[0], 1.0 / 1.01 * 1.0 / 1.02, &["DAI", "USDT"], &nodes);
            check_path!(&paths[1], 1.0 / 0.99, &["USDT"], &nodes);
            check_path!(&paths[2], 1.0 / 0.99 * 1.0 / 0.98, &["SOL", "USDT"], &nodes);
            check_path!(&paths[3], 1.0 / 0.95, &["USDT"], &nodes);
            assert_eq!(paths.len(), 4);
        }
        {
            let paths = get_paths("USDC", "USDT", 1000, 2);
            check_path!(
                &paths[2],
                1000.0 * 1.0 / 0.99 * 1.0 / 0.98 * 0.8 * 0.8,
                &["SOL", "USDT"],
                &nodes
            );
            check_path!(&paths[3], 1000.0 * 1.0 / 0.95 * 0.8, &["USDT"], &nodes);
            check_path!(
                &paths[1],
                1000.0 * (1.0 / 1.01) * (1.0 / 1.02) * 0.8 * 0.8,
                &["DAI", "USDT"],
                &nodes
            );
            check_path!(&paths[0], 1000.0 * 1.0 / 0.99 * 0.5, &["USDT"], &nodes);
            assert_eq!(paths.len(), 4);
        }
    }

    #[test]
    fn find_best_paths_performance_test() {
        panic_after(Duration::from_secs(3), || {
            let instant = Instant::now();
            let edges: Vec<(&str, &str, f64)> = (1..250)
                .map(|i| {
                    let d = (i as f64) / 10_000f64;
                    vec![
                        ("USDC", "USDT", 0.99 - d),
                        ("USDC", "USDT", 0.95 - d),
                        ("USDT", "USDC", 0.95 - d),
                        ("USDC", "SOL", (0.99 - d) / 200.0),
                        ("INF", "SOL", 1.03 - d),
                        ("SOL", "INF", 0.97 - d),
                        ("SOL", "USDT", (0.98 - d) * 200.0),
                        ("USDC", "TBTC", (0.99 - d) / 60_000.0),
                        ("TBTC", "USDT", (0.98 - d) * 60_000.0),
                        ("USDC", "wETH", (0.99 - d) / 4_000.0),
                        ("wETH", "USDT", (0.98 - d) * 4_000.0),
                        ("wETH", "SOL", (0.988 - d) * 4_000.0 / 200.0),
                        ("SOL", "wETH", (0.983 - d) * 4_000.0 * 200.0),
                        ("USDC", "DAI", 1.01 + d),
                        ("DAI", "USDT", 1.02 + d),
                        ("SOL", "JupSOL", 0.98 - d),
                        ("JupSOL", "USDT", 205.0 + d),
                    ]
                })
                .flatten()
                .collect();

            let (nodes, out_edges_per_node, edge_prices) = make_graph(edges.as_slice());

            let edge_price_fn = |edge: EdgeIndex, in_amount| {
                let price = if in_amount > 100 {
                    edge_prices[edge.idx()] * 0.8
                } else {
                    edge_prices[edge.idx()]
                };
                Some(EdgeInfo { price, accounts: 0 })
            };

            let get_paths = |from: &str, to: &str, in_amount, max_length| {
                best_price_paths_depth_search(
                    *nodes.get(&from.to_string()).unwrap(),
                    in_amount,
                    max_length,
                    64,
                    &out_edges_per_node,
                    &mut alloc_best_paths_by_node_for_test(
                        out_edges_per_node.len(),
                        10,
                        SwapMode::ExactIn,
                    ),
                    &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                    edge_price_fn,
                    &HashSet::new(),
                    false,
                    SwapMode::ExactIn,
                )
                .unwrap()[*nodes.get(&to.to_string()).unwrap()]
                .clone()
            };

            for _i in 0..10 {
                let paths = get_paths("INF", "USDC", 1, 5);
                assert_eq!(paths.is_empty(), false);
            }
            println!("Time taken : {} ms", instant.elapsed().as_millis());
        });
    }

    #[test]
    fn find_best_paths_performance_test_exact_out() {
        panic_after(Duration::from_secs(3), || {
            let instant = Instant::now();
            let edges: Vec<(&str, &str, f64)> = (1..250)
                .map(|i| {
                    let d = (i as f64) / 10_000f64;
                    vec![
                        ("USDC", "USDT", 0.99 - d),
                        ("USDC", "USDT", 0.95 - d),
                        ("USDT", "USDC", 0.95 - d),
                        ("USDC", "SOL", (0.99 - d) / 200.0),
                        ("INF", "SOL", 1.03 - d),
                        ("SOL", "INF", 0.97 - d),
                        ("SOL", "USDT", (0.98 - d) * 200.0),
                        ("USDC", "TBTC", (0.99 - d) / 60_000.0),
                        ("TBTC", "USDT", (0.98 - d) * 60_000.0),
                        ("USDC", "wETH", (0.99 - d) / 4_000.0),
                        ("wETH", "USDT", (0.98 - d) * 4_000.0),
                        ("wETH", "SOL", (0.988 - d) * 4_000.0 / 200.0),
                        ("SOL", "wETH", (0.983 - d) * 4_000.0 * 200.0),
                        ("USDC", "DAI", 1.01 + d),
                        ("DAI", "USDT", 1.02 + d),
                        ("SOL", "JupSOL", 0.98 - d),
                        ("JupSOL", "USDT", 205.0 + d),
                    ]
                })
                .flatten()
                .collect();

            let (nodes, out_edges_per_node, edge_prices) = make_graph(edges.as_slice());

            let edge_price_fn = |edge: EdgeIndex, in_amount| {
                let price = if in_amount > 100 {
                    edge_prices[edge.idx()] * 0.8
                } else {
                    edge_prices[edge.idx()]
                };
                Some(EdgeInfo { price, accounts: 0 })
            };

            let get_paths = |from: &str, to: &str, in_amount, max_length| {
                best_price_paths_depth_search(
                    *nodes.get(&from.to_string()).unwrap(),
                    in_amount,
                    max_length,
                    64,
                    &out_edges_per_node,
                    &mut alloc_best_paths_by_node_for_test(
                        out_edges_per_node.len(),
                        10,
                        SwapMode::ExactOut,
                    ),
                    &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                    edge_price_fn,
                    &HashSet::new(),
                    false,
                    SwapMode::ExactOut,
                )
                .unwrap()[*nodes.get(&to.to_string()).unwrap()]
                .clone()
            };

            for _i in 0..10 {
                let paths = get_paths("INF", "USDC", 1, 5);
                assert_eq!(paths.is_empty(), false);
            }
            println!("Time taken : {} ms", instant.elapsed().as_millis());
        });
    }

    #[test]
    fn should_find_same_top_path_when_asking_for_5_or_for_50_bests() {
        // Doing a USDC->USDT
        // Should find:
        // - Direct A
        // - Direct B
        // - USDC -> SOL -> USDT (D+E)
        // - USDC -> SOL -> DAI -> USDT (D+F+I)
        // - USDC -> BONK -> SOL -> USDT (G+H+E)
        // - USDC -> BONK -> SOL -> DAI -> USDT (G+H+F+I)
        // - USDC -> BONK -> USDT (G+J)
        // - USDC -> BONK -> USDT (G+K)
        // - USDC -> BONK -> SOL -> USDT (L+H+E)
        // - USDC -> BONK -> SOL -> DAI -> USDT (L+H+F+I)
        // - USDC -> BONK -> USDT (L+J)
        // - USDC -> BONK -> USDT (L+K)

        let (nodes, out_edges_per_node, edge_prices) = make_graph(&[
            ("USDC", "USDT", 0.99),              // A /
            ("USDC", "USDT", 0.95),              // B /
            ("USDT", "USDC", 2.0), // C / never taken when starting from USDC: no cycles allowed
            ("USDC", "SOL", 0.99 / 200.0), // D /
            ("SOL", "USDT", 0.98 * 200.0), // E /
            ("SOL", "DAI", 1.05 * 200.0), // F /
            ("USDC", "BONK", 100_000.0), // G /
            ("BONK", "SOL", 205.0 / 100_000.0), // H /
            ("DAI", "USDT", 0.97), // I /
            ("BONK", "USDT", 1.02 / 100_000.0), // J /
            ("BONK", "USDT", 1.025 / 100_000.0), // K /
            ("USDC", "BONK", 100_005.0), // L /
        ]);
        let edge_info_fn = |edge: EdgeIndex, _in_amount| {
            Some(EdgeInfo {
                price: edge_prices[edge.idx()],
                accounts: 7,
            })
        };

        let get_paths = |from: &str, to: &str, n_path, max_length| {
            // routing.find_best_route()
            best_price_paths_depth_search(
                *nodes.get(&from.to_string()).unwrap(),
                10000,
                max_length,
                64,
                &out_edges_per_node,
                &mut alloc_best_paths_by_node_for_test(
                    out_edges_per_node.len(),
                    n_path,
                    SwapMode::ExactIn,
                ),
                &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                edge_info_fn,
                &HashSet::new(),
                false,
                SwapMode::ExactIn,
            )
            .unwrap()[*nodes.get(&to.to_string()).unwrap()]
            .clone()
        };

        let paths1 = get_paths("USDC", "USDT", 5, 5);
        let paths2 = get_paths("USDC", "USDT", 50, 5);

        assert_eq!(paths1.len(), 5);
        assert_eq!(paths2.len(), 12);

        for i in 0..5 {
            assert_eq_f64!(paths1[i].0, paths2[i].0, 0.000001);
        }
    }

    #[test]
    fn should_find_same_top_path_when_asking_for_5_or_for_50_bests_for_exact_out() {
        // Doing a USDC->USDT
        // Should find:
        // - Direct A
        // - Direct B
        // - USDC -> SOL -> USDT (D+E)
        // - USDC -> SOL -> DAI -> USDT (D+F+I)
        // - USDC -> BONK -> SOL -> USDT (G+H+E)
        // - USDC -> BONK -> SOL -> DAI -> USDT (G+H+F+I)
        // - USDC -> BONK -> USDT (G+J)
        // - USDC -> BONK -> USDT (G+K)
        // - USDC -> BONK -> SOL -> USDT (L+H+E)
        // - USDC -> BONK -> SOL -> DAI -> USDT (L+H+F+I)
        // - USDC -> BONK -> USDT (L+J)
        // - USDC -> BONK -> USDT (L+K)

        let (nodes, out_edges_per_node, edge_prices) = make_graph(&[
            ("USDC", "USDT", 0.99),              // A /
            ("USDC", "USDT", 0.95),              // B /
            ("USDT", "USDC", 2.0), // C / never taken when starting from USDC: no cycles allowed
            ("USDC", "SOL", 0.99 / 200.0), // D /
            ("SOL", "USDT", 0.98 * 200.0), // E /
            ("SOL", "DAI", 1.05 * 200.0), // F /
            ("USDC", "BONK", 100_000.0), // G /
            ("BONK", "SOL", 205.0 / 100_000.0), // H /
            ("DAI", "USDT", 0.97), // I /
            ("BONK", "USDT", 1.02 / 100_000.0), // J /
            ("BONK", "USDT", 1.025 / 100_000.0), // K /
            ("USDC", "BONK", 100_005.0), // L /
        ]);
        let edge_info_fn = |edge: EdgeIndex, _in_amount| {
            Some(EdgeInfo {
                price: edge_prices[edge.idx()],
                accounts: 7,
            })
        };

        let get_paths = |from: &str, to: &str, n_path, max_length| {
            // routing.find_best_route()
            best_price_paths_depth_search(
                *nodes.get(&from.to_string()).unwrap(),
                10000,
                max_length,
                64,
                &out_edges_per_node,
                &mut alloc_best_paths_by_node_for_test(
                    out_edges_per_node.len(),
                    n_path,
                    SwapMode::ExactOut,
                ),
                &mut alloc_best_by_node_for_test(out_edges_per_node.len()),
                edge_info_fn,
                &HashSet::new(),
                false,
                SwapMode::ExactOut,
            )
            .unwrap()[*nodes.get(&to.to_string()).unwrap()]
            .clone()
        };

        let paths1 = get_paths("USDC", "USDT", 5, 5);
        let paths2 = get_paths("USDC", "USDT", 50, 5);

        assert_eq!(paths1.len(), 5);
        assert_eq!(paths2.len(), 12);

        for i in 0..5 {
            assert_eq_f64!(paths1[i].0, paths2[i].0, 0.000001);
        }
    }

    #[test]
    fn should_find_best_exact_in_route_fully_integrated() {
        let usdc = Pubkey::new_unique();
        let sol = Pubkey::new_unique();
        let mngo = Pubkey::new_unique();
        let pool_1 = Pubkey::new_unique();
        let pool_2 = Pubkey::new_unique();
        let pool_3 = Pubkey::new_unique();

        //
        let chain_data = Arc::new(ChainDataAccountProvider::new(ChainDataArcRw::new(
            Default::default(),
        ))) as AccountProviderView;
        let dex = Arc::new(MockDexInterface {}) as Arc<dyn DexInterface>;
        let edges = vec![
            Arc::new(make_edge(
                &dex,
                &pool_1,
                &usdc,
                &sol,
                &chain_data,
                6,
                1.0,
                1.0 / 0.1495,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_1,
                &sol,
                &usdc,
                &chain_data,
                9,
                150.0,
                0.1497,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_2,
                &usdc,
                &sol,
                &chain_data,
                6,
                1.0,
                1.0 / 0.1498,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_2,
                &sol,
                &usdc,
                &chain_data,
                9,
                150.0,
                0.1501,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_3,
                &usdc,
                &mngo,
                &chain_data,
                6,
                1.00,
                1.0 / 0.0198,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_3,
                &mngo,
                &usdc,
                &chain_data,
                6,
                0.02,
                0.0197,
            )),
        ];
        let pwa = vec![100, 1000];
        let config = Config {
            ..Config::default()
        };

        let routing = Routing::new(&config, pwa, edges);

        let path = routing
            .find_best_route(
                &chain_data,
                &sol,
                &mngo,
                1_000_000_000,
                40,
                true,
                &Default::default(),
                None,
                SwapMode::ExactIn,
            )
            .unwrap();

        assert_eq!(2, path.steps.len());
        assert_eq!(pool_2, path.steps[0].edge.id.key());
        assert_eq!(pool_3, path.steps[1].edge.id.key());
        assert_eq!(7580808080, path.out_amount);
        assert_eq_f64!(
            1_000_000_000.0 * 0.1501 * 1.0 / 0.0198,
            path.out_amount as f64,
            1.0
        );
    }

    #[test]
    fn should_find_best_exact_in_route_fully_integrated_exact_out() {
        let usdc = Pubkey::new_unique();
        let sol = Pubkey::new_unique();
        let mngo = Pubkey::new_unique();
        let pool_1 = Pubkey::new_unique();
        let pool_2 = Pubkey::new_unique();
        let pool_3 = Pubkey::new_unique();

        //
        let chain_data = Arc::new(ChainDataAccountProvider::new(ChainDataArcRw::new(
            Default::default(),
        ))) as AccountProviderView;
        let dex = Arc::new(MockDexInterface {}) as Arc<dyn DexInterface>;
        let edges = vec![
            Arc::new(make_edge(
                &dex,
                &pool_1,
                &usdc,
                &sol,
                &chain_data,
                6,
                1.0,
                1.0 / 0.1495,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_1,
                &sol,
                &usdc,
                &chain_data,
                9,
                150.0,
                0.1497,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_2,
                &usdc,
                &sol,
                &chain_data,
                6,
                1.0,
                1.0 / 0.1498,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_2,
                &sol,
                &usdc,
                &chain_data,
                9,
                150.0,
                0.1501,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_3,
                &usdc,
                &mngo,
                &chain_data,
                6,
                1.00,
                1.0 / 0.0198,
            )),
            Arc::new(make_edge(
                &dex,
                &pool_3,
                &mngo,
                &usdc,
                &chain_data,
                6,
                0.02,
                0.0197,
            )),
        ];
        let pwa = vec![100, 1000];
        let config = Config {
            ..Config::default()
        };

        let routing = Routing::new(&config, pwa, edges);

        let path = routing
            .find_best_route(
                &chain_data,
                &sol,
                &mngo,
                1_000_000_000,
                40,
                true,
                &Default::default(),
                None,
                SwapMode::ExactOut,
            )
            .unwrap();

        assert_eq!(2, path.steps.len());
        assert_eq!(pool_2, path.steps[0].edge.id.key());
        assert_eq!(pool_3, path.steps[1].edge.id.key());
        assert_eq!(131_912_059, path.in_amount);
        assert_eq!(1_000_000_000, path.out_amount);
        assert_eq_f64!(
            1_000_000_000.0 * 0.0198 * 1.0 / 0.1501,
            path.in_amount as f64,
            1.0
        );
    }

    fn make_edge(
        dex: &Arc<dyn DexInterface>,
        key: &Pubkey,
        input_mint: &Pubkey,
        output_mint: &Pubkey,
        chain_data: &AccountProviderView,
        decimals: u8,
        input_price_usd: f64,
        pool_price: f64,
    ) -> Edge {
        let edge = Edge {
            input_mint: input_mint.clone(),
            output_mint: output_mint.clone(),
            dex: dex.clone(),
            id: Arc::new(MockDexIdentifier {
                key: key.clone(),
                input_mint: input_mint.clone(),
                output_mint: output_mint.clone(),
                price: pool_price,
            }),
            accounts_needed: 10,
            state: Default::default(),
        };

        edge.update_internal(chain_data, decimals, input_price_usd, &vec![100, 1000]);
        edge
    }

    fn panic_after<T, F>(d: Duration, f: F) -> T
    where
        T: Send + 'static,
        F: FnOnce() -> T,
        F: Send + 'static,
    {
        let (done_tx, done_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let val = f();
            done_tx.send(()).expect("Unable to send completion signal");
            val
        });

        match done_rx.recv_timeout(d) {
            Ok(_) => handle.join().expect("Thread panicked"),
            Err(mpsc::RecvTimeoutError::Timeout) => panic!("Thread took too long"),
            Err(_) => panic!("Something went wrong"),
        }
    }
}
