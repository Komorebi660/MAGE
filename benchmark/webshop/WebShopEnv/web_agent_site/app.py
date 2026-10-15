import argparse, random, threading
from pathlib import Path
from ast import literal_eval
import itertools
import os
from flask import (
    Flask,
    request,
    redirect,
    url_for
)

from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    convert_web_app_string_to_var,
    get_top_n_product_from_keywords,
    get_product_per_page,
    map_action_to_html,
    END_BUTTON,
    PRODUCT_WINDOW
)
from web_agent_site.engine.goal import get_reward, get_goals, get_bundle_goals
from web_agent_site.utils import (
    generate_mturk_code,
    DEFAULT_FILE_PATH,
    DEBUG_PROD_SIZE,
    BUNDLE_GOAL_PATH,
)
app = Flask(__name__)

search_engine = None
all_products = None
product_item_dict = None
product_prices = None
attribute_to_asins = None
goals = None
bundle_goals = None
weights = None

user_sessions = dict()
_sessions_lock = threading.Lock()
_init_lock = threading.Lock()
user_log_dir = None
subreward_log_dir = None
SHOW_ATTRS_TAB = False


def _init_session_rewards(sess):
    """Initialise per-session reward tracking fields."""
    sess.setdefault('s_s', 0.0)
    sess.setdefault('i_s', 0.0)
    sess.setdefault('b_s', 0.0)
    sess.setdefault('sub_reward', 0.0)


@app.route('/')
def home():
    return redirect(url_for('index', session_id="abc"))

@app.route('/<session_id>', methods=['GET', 'POST'])
def index(session_id):
    global user_log_dir
    global subreward_log_dir
    global all_products, product_item_dict, \
           product_prices, attribute_to_asins, \
           search_engine, \
           goals, bundle_goals, weights, user_sessions

    with _init_lock:
        if search_engine is None:
            all_products, product_item_dict, product_prices, attribute_to_asins = \
                load_products(
                    filepath=DEFAULT_FILE_PATH,
                    num_products=DEBUG_PROD_SIZE
                )
            search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
            goals = get_goals(all_products, product_prices)
            weights = [goal['weight'] for goal in goals]
            # Load bundle goals if the file exists
            if os.path.exists(BUNDLE_GOAL_PATH):
                bundle_goals = get_bundle_goals(BUNDLE_GOAL_PATH, product_item_dict)
    
    with _sessions_lock:
        if session_id not in user_sessions and session_id.startswith('bundle_'):
            # Session format: bundle_{task_id}_{sub_idx}[_w{worker_id}]
            parts = session_id[len('bundle_'):].split('_')
            task_id, sub_idx = int(parts[0]), int(parts[1])
            goal_key = f"{task_id}_{sub_idx}"
            if bundle_goals and goal_key in bundle_goals:
                goal = bundle_goals[goal_key]
            else:
                raise ValueError(f"Bundle goal not found for session {session_id}, key={goal_key}")
            instruction_text = goal['instruction_text']
            sess = {'goal': goal, 'done': False}
            _init_session_rewards(sess)
            user_sessions[session_id] = sess
        elif session_id not in user_sessions and 'fixed' in session_id:
            # session format: fixed_{goal_idx}
            goal_dix = int(session_id.split('_')[-1])
            goal = goals[goal_dix]
            instruction_text = goal['instruction_text']
            sess = {'goal': goal, 'done': False}
            _init_session_rewards(sess)
            user_sessions[session_id] = sess
        elif session_id not in user_sessions:
            goal = random.choices(goals, weights)[0]
            instruction_text = goal['instruction_text']
            sess = {'goal': goal, 'done': False}
            _init_session_rewards(sess)
            user_sessions[session_id] = sess
        else:
            instruction_text = user_sessions[session_id]['goal']['instruction_text']

    if request.method == 'POST' and 'search_query' in request.form:
        keywords = request.form['search_query'].lower().split(' ')
        return redirect(url_for(
            'search_results',
            session_id=session_id,
            keywords=keywords,
            page=1,
        ))

    sess = user_sessions[session_id]
    if request.method == 'GET':
        sess['sub_reward'] = sess['s_s'] / 3.0 + sess['i_s'] / 3.0 + sess['b_s'] / 3.0

    return map_action_to_html(
        'start',
        session_id=session_id,
        instruction_text=instruction_text,
        progress_reward=sess['sub_reward']
    )


@app.route(
    '/search_results/<session_id>/<keywords>/<page>',
    methods=['GET', 'POST']
)
def search_results(session_id, keywords, page):
    instruction_text = user_sessions[session_id]['goal']['instruction_text']
    page = convert_web_app_string_to_var('page', page)
    keywords = convert_web_app_string_to_var('keywords', keywords)
    top_n_products = get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins,
    )
    # Get product list from search result asins and get list of corresponding URLs
    products = get_product_per_page(top_n_products, page)

    if request.method == 'GET':
        # calculate sub-reward (search recall reward)
        goal = user_sessions[session_id]['goal']
        search_max = 0.0
        maxr = False
        for i in range(len(products)):
            product_r = []
            single_product = products[i]
            all_options = list(single_product['options'].values())
            all_combinations = list(itertools.product(*all_options))
            if len(all_combinations) > 0:
                for c in all_combinations:
                    single_product_asin = single_product['asin']
                    price = product_prices[single_product_asin]
                    reward, _ = get_reward(single_product, goal, price, dict(enumerate(c)),
                                           verbose=True)
                    product_r.append(reward)
                    if reward > search_max:
                        search_max = reward
                    if reward == 1.0:
                        maxr = True
                        search_max = reward

            else:
                price = single_product['pricing'][0]
                reward, _ = get_reward(single_product, goal, price, [], verbose=True)
                if reward == 1.0:
                    maxr = True
                if reward > search_max:
                    search_max = reward
        sess = user_sessions[session_id]
        if search_max > sess['s_s']:
            sess['s_s'] = search_max
        sess['sub_reward'] = sess['s_s'] / 3.0 + sess['i_s'] / 3.0 + sess['b_s'] / 3.0



    html = map_action_to_html(
        'search',
        session_id=session_id,
        products=products,
        keywords=keywords,
        page=page,
        total=(len(top_n_products) + PRODUCT_WINDOW - 1) // PRODUCT_WINDOW,
        instruction_text=instruction_text,
        progress_reward=user_sessions[session_id]['sub_reward']
    )
    return html


@app.route(
    '/item_page/<session_id>/<asin>/<keywords>/<page>/<options>',
    methods=['GET', 'POST']
)
def item_page(session_id, asin, keywords, page, options):
    options = literal_eval(options)
    product_info = dict(product_item_dict[asin])
    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction

    if request.method == 'GET':
        # calculate sub-reward (item page score)
        goal = user_sessions[session_id]['goal']
        single_product = product_info
        all_options = list(single_product['options'].values())
        # generate all possible combinations of options
        all_combinations = list(itertools.product(*all_options))
        if len(all_combinations) > 0:
            for c in all_combinations:
                maxr = -1  # whether contains goal item in item page
                price = product_prices[asin]
                reward, _ = get_reward(single_product, goal, price, dict(enumerate(c)),
                                       verbose=True)
                if reward > maxr:
                    maxr = reward
                    if reward == 1.0:
                        break
        else:
            price = single_product['pricing'][0]
            reward, _ = get_reward(single_product, goal, price, [], verbose=True)
            maxr = reward

        sess = user_sessions[session_id]
        if maxr > sess['i_s']:
            sess['i_s'] = maxr
        sess['sub_reward'] = sess['s_s'] / 3.0 + sess['i_s'] / 3.0 + sess['b_s'] / 3.0


    html = map_action_to_html(
        'click',
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        show_attrs=SHOW_ATTRS_TAB,
        progress_reward=user_sessions[session_id]['sub_reward']
    )
    return html


@app.route(
    '/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>',
    methods=['GET', 'POST']
)
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    options = literal_eval(options)
    product_info = dict(product_item_dict[asin])

    goal_instruction = user_sessions[session_id]['goal']['instruction_text']
    product_info['goal_instruction'] = goal_instruction

    if request.method == 'GET':
        # calculate sub-reward (item page score)
        goal = user_sessions[session_id]['goal']
        single_product = product_info
        all_options = list(single_product['options'].values())
        # generate all possible combinations of options
        all_combinations = list(itertools.product(*all_options))
        if len(all_combinations) > 0:
            for c in all_combinations:
                maxr = -1  # whether contains goal item in item page
                price = product_prices[asin]
                reward, _ = get_reward(single_product, goal, price, dict(enumerate(c)),
                                       verbose=True)
                if reward > maxr:
                    maxr = reward
                    if reward == 1.0:
                        break
        else:
            price = single_product['pricing'][0]
            reward, _ = get_reward(single_product, goal, price, [], verbose=True)
            maxr = reward


        sess = user_sessions[session_id]
        if maxr > sess['i_s']:
            sess['i_s'] = maxr
        sess['sub_reward'] = sess['s_s'] / 3.0 + sess['i_s'] / 3.0 + sess['b_s'] / 3.0


    html = map_action_to_html(
        f'click[{sub_page}]',
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        progress_reward=user_sessions[session_id]['sub_reward']
    )
    return html


@app.route('/done/<session_id>/<asin>/<options>', methods=['GET', 'POST'])
def done(session_id, asin, options):
    options = literal_eval(options)
    goal = user_sessions[session_id]['goal']
    purchased_product = product_item_dict[asin]
    price = product_prices[asin]

    reward, reward_info = get_reward(
        purchased_product,
        goal,
        price=price,
        options=options,
        verbose=True
    )
    sess = user_sessions[session_id]
    sess['done'] = True
    sess['reward'] = reward

    sess['b_s'] = reward
    last_sub_reward = sess['s_s'] / 3.0 + sess['i_s'] / 3.0 + sess['b_s'] / 3.0
    # Reset per-session reward trackers (session is done).
    sess['s_s'] = 0.0
    sess['i_s'] = 0.0
    sess['b_s'] = 0.0
    sess['sub_reward'] = 0.0


    return map_action_to_html(
        f'click[{END_BUTTON}]',
        session_id=session_id,
        reward=reward,
        asin=asin,
        options=options,
        reward_info=reward_info,
        query=purchased_product['query'],
        category=purchased_product['category'],
        product_category=purchased_product['product_category'],
        goal_attrs=sess['goal']['attributes'],
        purchased_attrs=purchased_product['Attributes'],
        goal=goal,
        mturk_code=generate_mturk_code(session_id),
        progress_reward=last_sub_reward
    )

@app.route('/failed/<session_id>', methods=['HEAD'])
def failed(session_id):
    if session_id in user_sessions:
        sess = user_sessions[session_id]
        sess['b_s'] = 0.0
        sess['s_s'] = 0.0
        sess['i_s'] = 0.0
        sess['sub_reward'] = 0.0
    return ''


# Keep the old route for backward compatibility (no-op without session context).
@app.route('/failed', methods=['HEAD'])
def failed_legacy():
    return ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebShop flask app backend configuration")
    parser.add_argument("--log", action='store_true', help="Log actions on WebShop in trajectory file")
    parser.add_argument("--attrs", action='store_true', help="Show attributes tab in item page")

    args = parser.parse_args()
    if args.log:
        subreward_log_dir = Path('subreward_log')
        subreward_log_dir.mkdir(parents=True, exist_ok=True)
    SHOW_ATTRS_TAB = args.attrs

    app.run(host='0.0.0.0', port=3000)
