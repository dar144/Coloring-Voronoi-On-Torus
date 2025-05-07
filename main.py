# Source: (C++) http://www.cs.hmc.edu/~mbrubeck/voronoi.html

import heapq
import itertools
import random
import networkx as nx
import math


class Point:
    def __init__(self, x, y):
       self.x = x
       self.y = y


class Event:
    def __init__(self, x = 0.0, p = None, a = None):
        self.x = x   # key 
        self.p = p   # point
        self.a = a   # arc
        self.valid = True

class Arc:    
    def __init__(self, p, a=None, b=None):
        self.p = p     # site
        self.pprev = a 
        self.pnext = b 
        self.e = None     # event
        self.s0 = None    # left edge 
        self.s1 = None    # right edge

class Edge:
    def __init__(self, p = None):
        self.start = p     # start point
        self.end = None    # end point
        self.done = False  # is finished

    def finish(self, p):
        if self.done: return
        self.end = p
        self.done = True   

class Graph:
    def __init__(self):
        self.vertices = {} 
        self.edges = [] 

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def push(self, item):
        # check for duplicate
        if item in self.entry_finder: return
        count = next(self.counter)
        
        # use x-coordinate as a primary key (heapq in python is min-heap)
        entry = [item.x, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)


    def pop(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def top(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                self.push(item)
                return item
        raise KeyError('top from an empty priority queue')

    def empty(self):
        return not self.pq
    

class Voronoi:
    def __init__(self, points):
        self.output = [] # list of line Edge
        self.arc = None  # binary tree for parabola arcs
        self.sites = points

        self.points = PriorityQueue() # site events
        self.event = PriorityQueue() # circle events
        self.areas = {}
        self.graph = Graph()
        # self.test = {}

        # bounding box
        self.x0 = -50.0
        self.x1 = -50.0
        self.y0 = 550.0
        self.y1 = 550.0

        # insert points to site event
        for pts in points:
            point = Point(pts[0], pts[1])
            self.points.push(point)

            if point.x < self.x0: self.x0 = point.x
            if point.y < self.y0: self.y0 = point.y
            if point.x > self.x1: self.x1 = point.x
            if point.y > self.y1: self.y1 = point.y


        dx = (self.x1 - self.x0 + 1) / 5.0
        dy = (self.y1 - self.y0 + 1) / 5.0
        self.x0 = self.x0 - dx
        self.x1 = self.x1 + dx
        self.y0 = self.y0 - dy
        self.y1 = self.y1 + dy


    def process(self):
        while not self.points.empty():
            if not self.event.empty() and (self.event.top().x <= self.points.top().x):
                self.process_event() # handle circle event
            else:
                self.process_point() # handle site event

        # after all points, process remaining circle events
        while not self.event.empty():
            self.process_event()

        self.finish_edges()

    def process_point(self):
        # get next event from site pq
        p = self.points.pop()
        # add new arc 
        self.arc_insert(p)
        

    def process_event(self):
        # get next event from circle pq
        e = self.event.pop()
        

        if e.valid:
            # start new edge
            s = Edge(e.p)
            self.output.append(s)

            # add vertices to a list
            vertices_length = len(self.graph.vertices)
            if e.p not in self.graph.vertices:
                self.graph.vertices[(e.p.x, e.p.y)] = vertices_length

            a = e.a

            # finish the edges before and after a
            if a.s0 is not None: 
                a.s0.finish(e.p)
            if a.s1 is not None: 
                a.s1.finish(e.p)


            if a.pprev is not None: self.check_circle_event(a.pprev, e.x)
            if a.pnext is not None: self.check_circle_event(a.pnext, e.x)

    def arc_insert(self, p):
        if self.arc is None:
            self.arc = Arc(p)
        else:
            i = self.arc
            while i is not None:
                flag, z = self.intersect(p, i)
                if flag:
                    # new parabola intersects arc i
                    flag, zz = self.intersect(p, i.pnext)
                    if (i.pnext is not None) and (not flag):
                        i.pnext.pprev = Arc(i.p, i, i.pnext)
                        i.pnext = i.pnext.pprev
                    else:
                        i.pnext = Arc(i.p, i)
                    i.pnext.s1 = i.s1

                    # add p between i and i.pnext
                    i.pnext.pprev = Arc(p, i, i.pnext)
                    i.pnext = i.pnext.pprev

                    i = i.pnext 

                    # add new half-edges connected to i's endpoints
                    seg = Edge(z)
                    self.output.append(seg)

                    i.pprev.s1 = i.s0 = seg

                    seg = Edge(z)
                    self.output.append(seg)

                    i.pnext.s0 = i.s1 = seg

                    # check for new circle events around the new arc
                    self.check_circle_event(i, p.x)
                    self.check_circle_event(i.pprev, p.x)
                    self.check_circle_event(i.pnext, p.x)

                    return
                        
                i = i.pnext

            # if p never intersects an arc, append it to the list
            i = self.arc
            while i and i.pnext is not None:
                i = i.pnext
            i.pnext = Arc(p, i)
            
            # insert new Edge between p and i
            x = self.x0
            y = (i.pnext.p.y + i.p.y) / 2.0
            start = Point(x, y)

            seg = Edge(start)
            i.s1 = i.pnext.s0 = seg
            self.output.append(seg)

    def check_circle_event(self, i, x0):
        # look for a new circle event for arc i
        if (i.e is not None) and (i.e.x  != self.x0):
            i.e.valid = False
        i.e = None

        if (i.pprev is None) or (i.pnext is None): return

        flag, x, o = self.circle(i.pprev.p, i.p, i.pnext.p)
        if flag and (x > self.x0):
            i.e = Event(x, o, i)
            self.event.push(i.e)


    def circle(self, a, b, c):
        if ((b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y)) > 0: return False, None, None

        A = b.x - a.x
        B = b.y - a.y
        C = c.x - a.x
        D = c.y - a.y
        E = A*(a.x + b.x) + B*(a.y + b.y)
        F = C*(a.x + c.x) + D*(a.y + c.y)
        G = 2*(A*(c.y - b.y) - B*(c.x - b.x))

        if (G == 0): return False, None, None 

        ox = 1.0 * (D*E - B*F) / G
        oy = 1.0 * (A*F - C*E) / G

        # o.x plus radius equals max x coord
        x = ox + math.sqrt((a.x-ox)**2 + (a.y-oy)**2)
        o = Point(ox, oy)
           
        return True, x, o
        
    def intersect(self, p, i):
        # check whether a new parabola at point p intersect with arc i
        if (i is None): return False, None
        if (i.p.x == p.x): return False, None

        a = 0.0
        b = 0.0

        if i.pprev is not None:
            a = (self.intersection(i.pprev.p, i.p, 1.0*p.x)).y
        if i.pnext is not None:
            b = (self.intersection(i.p, i.pnext.p, 1.0*p.x)).y

        if (((i.pprev is None) or (a <= p.y)) and ((i.pnext is None) or (p.y <= b))):
            py = p.y
            px = 1.0 * ((i.p.x)**2 + (i.p.y-py)**2 - p.x**2) / (2*i.p.x - 2*p.x)
            res = Point(px, py)
            return True, res
        return False, None

    def intersection(self, p0, p1, l):
        # get the intersection of two parabolas
        p = p0
        if (p0.x == p1.x):
            py = (p0.y + p1.y) / 2.0
        elif (p1.x == l):
            py = p1.y
        elif (p0.x == l):
            py = p0.y
            p = p1
        else:
            # use quadratic formula
            z0 = 2.0 * (p0.x - l)
            z1 = 2.0 * (p1.x - l)

            a = 1.0/z0 - 1.0/z1;
            b = -2.0 * (p0.y/z0 - p1.y/z1)
            c = 1.0 * (p0.y**2 + p0.x**2 - l**2) / z0 - 1.0 * (p1.y**2 + p1.x**2 - l**2) / z1

            py = 1.0 * (-b-math.sqrt(b*b - 4*a*c)) / (2*a)
            
        px = 1.0 * (p.x**2 + (p.y-py)**2 - l**2) / (2*p.x-2*l)
        res = Point(px, py)
        return res

    def finish_edges(self):
        l = self.x1 + (self.x1 - self.x0) + (self.y1 - self.y0)
        i = self.arc
        while i and i.pnext is not None:
            if i.s1 is not None:
                p = self.intersection(i.p, i.pnext.p, l*2.0)
                i.s1.finish(p)
            i = i.pnext

    def print_output(self):
        it = 0
        for o in self.output:
            it = it + 1
            p0 = o.start
            p1 = o.end
            print (p0.x, p0.y, p1.x, p1.y)

    def get_output(self):
        res = []
        for o in self.output:
            if o.end:
                p0 = o.start
                p1 = o.end
                res.append((p0.x, p0.y, p1.x, p1.y))
        return res


def get_points_with_pbc(width, height, points_init):
    points_pbc = []
    points_help = []
    for point in points_init:
        
        x, y = point
        if y <= height / 2:
            points_pbc.append((x, y + height))
            if x >= width / 2:
                points_help.append((x - width, y + height))
            else:
                points_help.append((x + width, y + height)) 
        else:
            points_pbc.append((x, y - height))

        if x <= width / 2:    
            points_pbc.append((x + width, y))
            if x >= width / 2:
                points_help.append((x - width, y - height))
            else:
                points_help.append((x + width, y - height)) 
        else:
            points_pbc.append((x - width, y))

    return points_init + points_pbc + points_help


def set_graph_edges(lines, vp):
    # Add graph edges
    i = 0
    while i < len(lines) - 1:
        p1 = (lines[i][0], lines[i][1])
        p2 = (lines[i][2], lines[i][3])
        p1_next = (lines[i+1][0], lines[i+1][1])
        p2_next = (lines[i+1][2], lines[i+1][3])


        if p1 != p1_next:
            if p1 in vp.graph.vertices and p2 in vp.graph.vertices:
                vp.graph.edges.append((vp.graph.vertices[p1], vp.graph.vertices[p2]))
            i += 1
        else:
            if p2 in vp.graph.vertices and p2_next in vp.graph.vertices:
                vp.graph.edges.append((vp.graph.vertices[p2], vp.graph.vertices[p2_next]))
            i += 2

    # The last line
    lines_length = len(lines) - 1
    p1 = (lines[lines_length][0], lines[lines_length][1])
    p2 = (lines[lines_length][2], lines[lines_length][3])   
    if p1 in vp.graph.vertices and p2 in vp.graph.vertices:
                vp.graph.edges.append((vp.graph.vertices[p1], vp.graph.vertices[p2]))

def find_faces(embedding):
    faces = []
    visited_half_edges = set()

    def sort_by_inner_array_length(arrays):
        return sorted(arrays, key=len)

    for v in embedding:
        for w in embedding[v]:
            if (v, w) not in visited_half_edges:
                face = list(embedding.traverse_face(v, w, mark_half_edges=visited_half_edges))
                faces.append(face)

    sorted_arrays = sort_by_inner_array_length(faces)
    faces = sorted_arrays[:-1]
    return faces


def find_adjacent_cycles(faces):
    cycles = {}

    for i in range(len(faces)):
        cycles[i] = faces[i]

    def get_edges(vertices):
        edges = set()
        for i in range(len(vertices)):
            edge = tuple(sorted([vertices[i], vertices[(i + 1) % len(vertices)]]))  # Упорядоченные рёбра
            edges.add(edge)
        return edges

    cycle_edges = {cycle_id: get_edges(vertices) for cycle_id, vertices in cycles.items()}

    adjacency = {cycle_id: set() for cycle_id in cycles}

    for cycle_a, edges_a in cycle_edges.items():
        for cycle_b, edges_b in cycle_edges.items():
            if cycle_a != cycle_b and edges_a & edges_b: 
                adjacency[cycle_a].add(cycle_b)
                adjacency[cycle_b].add(cycle_a)

    return adjacency

def DSatur(graph):
    colors = {} 
    saturation = {node: 0 for node in graph}  
    degrees = {node: len(neighbors) for node, neighbors in graph.items()}  
    uncolored = set(graph.keys())  
    current = max(degrees, key=degrees.get)

    while uncolored:
        neighbor_colors = {colors[neighbor] for neighbor in graph[current] if neighbor in colors}
        assigned_color = next(color for color in range(len(graph)) if color not in neighbor_colors)
        colors[current] = assigned_color
        uncolored.remove(current)

        for neighbor in graph[current]:
            if neighbor in uncolored:
                neighbor_colors = {colors[n] for n in graph[neighbor] if n in colors}
                saturation[neighbor] = len(neighbor_colors)

        if uncolored:
            current = max(uncolored, key=lambda node: (saturation[node], degrees[node]))

    return colors

def RLF(graph):

    vertices = set(graph.keys()) 
    colors = {}  
    current_color = 0 
    
    while vertices:  
       
        start_vertex = max(vertices, key=lambda v: len(graph[v]))
        independent_set = {start_vertex} 
        
        for vertex in vertices - {start_vertex}: 
            if not any(neighbor in independent_set for neighbor in graph[vertex]):
                independent_set.add(vertex)
        
        for vertex in independent_set:
            colors[vertex] = current_color
        
        vertices -= independent_set
        current_color += 1  
    
    return colors

def greedy(graph):
    colors = {}  
    
    for vertex in graph:  
        neighbor_colors = {colors[neighbor] for neighbor in graph[vertex] if neighbor in colors}
        
        vertex_color = 0
        while vertex_color in neighbor_colors:
            vertex_color += 1
        colors[vertex] = vertex_color
    
    return colors


if __name__ == '__main__':
    width, height = 500, 500
    num_of_points = [5]
    num_of_iter = 100000
    res = []
    not_planar = 0

    # points = [(random.randint(0, width), random.randint(0, height)) for _ in range(10)]
    points = [(141, 341), (117, 341), (158, 448), (20, 394), (182, 48)]
    points_ext = get_points_with_pbc(width, height, points)
    vp = Voronoi(points_ext)
    vp.process()
    lines = vp.get_output()
    print(len(lines))
    set_graph_edges(lines, vp)
    vp.draw_voronoi()


    for n in num_of_points:
        for _ in range(num_of_iter): 
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(n)]
            points_ext = get_points_with_pbc(width, height, points)
            
            vp = Voronoi(points_ext)
            vp.process()
            lines = vp.get_output()
            set_graph_edges(lines, vp)

            G = nx.Graph()
            G.add_edges_from(vp.graph.edges)
            # pos = nx.planar_layout(G)
            is_planar, embedding = nx.check_planarity(G)
            if not is_planar:
                not_planar += 1
            else:
                faces = find_faces(embedding)

                adjacency = find_adjacent_cycles(faces)

                coloring_DSatur = DSatur(adjacency)
                num_DSatur = len(set(coloring_DSatur.values()))

                coloring_RLF = RLF(adjacency)
                num_RLF = len(set(coloring_RLF.values()))

                coloring_greedy = greedy(adjacency)
                num_greedy = len(set(coloring_greedy.values()))


                res.append([num_greedy, num_DSatur, num_RLF])

    with open("output_5.txt", "w") as file:
        for lst in res:
            file.write(" ".join(map(str, lst)) + "\n")
