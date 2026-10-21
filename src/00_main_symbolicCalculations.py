from sympy import symbols, Matrix # noqa: I001
from sympy import cos, sin, tan, atan
from sympy import simplify
from sympy import latex
from sympy import lambdify
from sympy import solve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class SymbolicEquations:
    def __init__(self):
        self.f = symbols('f', positive=True)
        self.x_i = symbols('x^i_{vPOI}')
        self.c_x = symbols('c_x')
        self.y_i = symbols('y^i_{vPOI}')
        self.c_y = symbols('c_y')
        self.psi = symbols('\\psi')
        self.theta = symbols('\\theta', negative=True)
        self.n_c = symbols('n_c')
        self.e_c = symbols('e_c')
        self.d_c = symbols('d_c', negative=True)
        self.l = symbols('p', positive=True)
        self.A = symbols('A')
        self.B = symbols('B')
        self.C = symbols('C')
        self.mu_f = symbols('\\Delta{f}')
        self.mu_xi = symbols('\\Delta{x^i_{vPOI}}', positive=True)
        self.mu_cx = symbols('\\Delta{c_x}')
        self.mu_yi = symbols('\\Delta{y^i_{vPOI}}', positive=True)
        self.mu_cy = symbols('\\Delta{c_y}')
        self.mu_psi = symbols('\\Delta{\\psi}')
        self.mu_theta = symbols('\\Delta{\\theta}')
        self.mu_nc = symbols('\\Delta{n_c}')
        self.mu_ec = symbols('\\Delta{e_c}')
        self.mu_dc = symbols('\\Delta{d_c}')
        self.s = symbols('s', positive=True)

        self.exprA = self.f*sin(self.theta) + self.C*cos(self.theta)
        self.exprB = self.c_x - self.x_i
        self.exprC = self.c_y - self.y_i

        self.r_w = None
        self.r_w_subs = None

    def zeroUndesiredContributions(self, expr, exclude):
        contribution_symbols_list = [
            self.mu_f, self.mu_xi, self.mu_cx, self.mu_yi, self.mu_cy,
            self.mu_psi, self.mu_theta,
            self.mu_nc, self.mu_ec, self.mu_dc
        ]
        for s in contribution_symbols_list:
            if s not in exclude:
                expr = expr.subs(s, 0)
        expr = simplify(expr)
        return expr

    def constSizeObjectSubs(self, expr, direction):
        # Ensure the object is of a constant size s
        if direction == 'lateral':
            expr_B_lateral = self.s*self.f*cos(self.theta)/self.l
            expr = expr.subs(self.B, expr_B_lateral)
        elif direction == 'longitudinal':
            expr_C_longitudinal = self.s*self.f*(1 - cos(2*self.theta))/(2*self.d_c - self.s*sin(2*self.theta))
            expr = expr.subs(self.C, expr_C_longitudinal)
        else:
            raise ValueError("Direction must be either 'lateral' or 'longitudinal'")
        
        # Normalize by size
        expr = expr / self.s
        expr = simplify(expr)

        return expr

    def compute_SF_error(self, SF, SF_bias, activeError, direction, norm = "noNorm"):
        # MACRO SUBSTITUTIONS FOR SIMPLIFICATION
        # Orient such that north corrsponds to the longitudinal axis and east to the lateral axis
        SF = SF.subs(self.psi, 0).subs(self.c_x - self.x_i, 0).subs(self.c_y - self.y_i, 0)
        
        # Move the origin object to the center of the image
        SF_bias = SF_bias.subs(self.psi, 0).subs(self.c_x - self.x_i, 0).subs(self.c_y - self.y_i, 0)

        SF_bias = self.zeroUndesiredContributions(SF_bias, exclude=activeError)

        if norm == "bespoke":
            Err_SF = (SF_bias[0]**2 + SF_bias[1]**2)**0.5 - (SF[0]**2 + SF[1]**2)**0.5
        elif norm == "sympy":
            Err_SF = SF_bias.norm() - SF.norm()
        elif norm == "noNorm":
            Err_SF = SF_bias - SF
        else:
            raise ValueError("norm must be either 'bespoke', 'sympy' or 'noNorm'")

        Err_SF = self.constSizeObjectSubs(Err_SF, direction)

        return Err_SF

    def useDistanceThetaSubs(self, expr):
        return simplify(expr.subs(self.theta, atan(self.d_c / self.l)))

    def smallAngleApproximation(self, expr, angle_symbol):
        expr_smallAngle = expr.expand(trig=True)
        expr_smallAngle = expr_smallAngle.subs(sin(angle_symbol), angle_symbol).subs(cos(angle_symbol), 1)
        expr_smallAngle = simplify(expr_smallAngle)
        return expr_smallAngle
        col = {
            50: "r",
            85: "b",
            120: "g",
        }

        handles = []
        labels = []

        d_cs = [-50, -85, -120]
        _plot_subs = {
            self.d_c: -50,
            self.f: 21270,
            self.s: 2,
            self.mu_yi: 5000,
            self.mu_xi: 5000,
            self.mu_nc: 1,
            self.mu_ec: 1,
            self.mu_dc: 1,
            self.mu_theta: np.radians(0.1),  # radians
            self.mu_psi: np.radians(1),  # radians
            self.mu_f: 50
        }

        fig, (ax_longi, ax_lat) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

        for d_val in d_cs:
            _plot_subs[self.d_c] = d_val
        
            # Create a numeric function of l
            err_func_longi = lambdify(self.l, errSize_expr_longi.subs(_plot_subs), 'numpy')
            err_func_lat = lambdify(self.l, errSize_expr_lat.subs(_plot_subs), 'numpy')

            # Sample l between a small positive value and 400 (avoid division by zero)
            l_vals = np.linspace(0.1, 400, 1000)
            y_vals_longi = np.array([abs(err_func_longi(ls))*100 for ls in l_vals], dtype=float)
            y_vals_lat = np.array([abs(err_func_lat(ls))*100 for ls in l_vals], dtype=float)

            labels.append(f"Elevation {-d_val}m")
            h = ax_longi.plot(l_vals, y_vals_longi, lw=2, color=col[abs(d_val)], label=f"Elevation {-d_val}m")
            handles.append(h[0])
            ax_lat.plot(l_vals, y_vals_lat, lw=2, color=col[abs(d_val)], label=f"Elevation {-d_val}m")

        ax_longi.set_ylabel('size error [%]')
        ax_longi.set_title("Longitudinal Object")
        ax_longi.grid(True)
        ax_longi.set_xlim(0, 400)
        

        ax_lat.set_xlabel('l [m]')
        ax_lat.set_ylabel('size error [%]')
        ax_lat.set_title("Lateral Object")
        ax_lat.grid(True)
        ax_lat.set_xlim(0, 400)

        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.55, 0.01))
        fig.suptitle(name, fontweight='bold')
        fig.tight_layout(rect=[0, 0.04, 1, 1])  # Leave space at the bottom for the legend
        fig.savefig(f"{saveFolder}/{name}.pdf")

    def printAllLatex(self, filePath):
        with open(filePath, 'w') as f:
            # Localisation equations
            f.write("\\section{Localisation Equations}\n")
            f.write("\\begin{align*}\n")
            f.write("LF\\left([n_c\\ e_c\\ d_c\\ \\psi\\ \\theta\\ f\\ c_x\\ c_y\\ x^i_{vPOI}\\ y^i_{vPOI}\\ ]^T\\right) &=")
            f.write(f"{latex(Matrix(self.r_w_subs[0:2]))} \\\\\n")
            f.write(f"A = {latex(self.exprA)} \\\\\n")
            f.write(f"B = {latex(self.exprB)} \\\\\n")
            f.write(f"C = {latex(self.exprC)} \\\\\n")
            f.write("\\end{align*}\n")

            # Localisation error contributions
            f.write("\\section{Localisation Error Contributions}\n")
            f.write("\\begin{align*}\n")
            f.write(f"\\text{{Horizontal position error: }}&{latex(Matrix(self.err_pos[0:2]))} \\\\\n")
            f.write(f"\\text{{Vertical error: }}&{latex(Matrix(self.err_alt[0:2]))} = {latex(Matrix(self.err_alt_withDist[0:2]))} \\\\\n")
            f.write(f"\\text{{Attitude pitch error: }}&{latex(Matrix(self.err_pitch[0:2]))} \\approx {latex(Matrix(self.err_pitch_smallAngle[0:2]))} = {latex(Matrix(self.err_pitch_smallAngle_withDist[0:2]))} \\\\\n")
            f.write(f"\\text{{Attitude yaw error: }}&{latex(Matrix(self.err_yaw[0:2]))} \\approx {latex(Matrix(self.err_yaw_smallAngle[0:2]))} = {latex(Matrix(self.err_yaw_smallAngle_withDist[0:2]))} \\\\\n")
            f.write(f"\\text{{X image coordinate error: }}&{latex(Matrix(self.err_xi[0:2]))} = {latex(Matrix(self.err_xi_withDist[0:2]))} \\\\\n")
            f.write(f"\\text{{Y image coordinate error: }}&{latex(Matrix(self.err_yi[0:2]))} = {latex(Matrix(self.err_yi_withDist[0:2]))} \\\\\n")
            f.write("\\end{align*}\n")

            # Size error contributions
            f.write("\\section{Size Error Contributions}\n")
            f.write("\\begin{align*}\n")
            f.write(f"\\text{{Horizontal size error (longitudinal): }}&{latex(self.ErrSize__pos_long[0])} \\\\\n")
            f.write(f"\\text{{Horizontal size error (lateral): }}&{latex(self.ErrSize_pos_lat[1])} \\\\\n")
            f.write(f"\\text{{Vertical size error (longitudinal): }}&{latex(self.ErrSize_alt_long[0])} \\\\\n")
            f.write(f"\\text{{Vertical size error (lateral): }}&{latex(self.ErrSize_alt_lat[1])} = {latex(self.ErrSize_alt_lat_withDist[1])} \\\\\n")
            f.write(f"\\text{{Attitude pitch size error (longitudinal): }}&\\approx {latex(self.ErrSize_yaw_long_smallAngle_withDist[0])} \\approx {latex(self.ErrSize_yaw_long_smallAngle_withDist_smallObject[0])} \\\\\n")
            f.write(f"\\text{{Attitude pitch size error (lateral): }}&{latex(self.ErrSize_pitch_lat[1])} \\approx {latex(self.ErrSize_pitch_lat_smallAngle[1])} = {latex(self.ErrSize_pitch_lat_smallAngle_withDist[1])} \\\\\n")
            f.write(f"\\text{{Attitude yaw size error (longitudinal): }}&{latex(self.ErrSize_yaw_long)} \\\\\n")
            f.write(f"\\text{{Attitude yaw size error (lateral): }}&{latex(self.ErrSize_yaw_lat)} \\\\\n")
            f.write(f"\\text{{X image coordinate size error (longitudinal): }}&{latex(self.ErrSize_xi_long)} = {latex(self.ErrSize_xi_long_withDist)} \\\\\n")
            f.write(f"\\text{{X image coordinate size error (lateral): }}&{latex(self.ErrSize_xi_lat[1])} \\\\\n")
            f.write(f"\\text{{Y image coordinate size error (longitudinal): }}&= {latex(self.ErrSize_yi_long_withDist)} \\\\\n")
            f.write(f"\\text{{Y image coordinate size error (lateral): }}&{latex(self.ErrSize_yi_lat[1])} = {latex(self.ErrSize_yi_lat_withDist[1])} \\\\\n")
            f.write(f"\\text{{Focal length size error (longitudinal): }}&{latex(self.ErrSize_f_long[0])} = {latex(self.ErrSize_f_long_withDist[0])} \\approx {latex(self.ErrSize_f_long_withDist_smallObject[0])} \\\\\n")
            f.write(f"\\text{{Focal length size error (lateral): }}&{latex(self.ErrSize_f_lat[1])} = {latex(self.ErrSize_f_lat_withDist[1])} \\\\\n")
            f.write("\\end{align*}\n")
            
            # Equivalence relations
            f.write("\\section{Equivalence Relations}\n")
            f.write("\\begin{align*}\n")
            f.write(f"\\text{{Pitch and Y image equivalence: }}&{latex(self.err_pitch_yi_equivalence)} \\\\\n")
            f.write(f"\\text{{Yaw and X image equivalence: }}&{latex(self.err_yaw_xi_equivalence)} \\\\\n")
            f.write("\\end{align*}\n")

    def localisation(self):
        # Position of the object in the image (camera frame)
        r_c = Matrix([[self.f], [self.x_i - self.c_x], [self.y_i - self.c_y]])

        # Rotation matrix from camera to world (ZY Euler angles)
        C_cw = Matrix([
            [cos(self.psi)*cos(self.theta), -sin(self.psi), cos(self.psi)*sin(self.theta)],
            [sin(self.psi)*cos(self.theta),  cos(self.psi), sin(self.psi)*sin(self.theta)],
            [        -sin(self.theta),           0,           cos(self.theta)]
        ])

        # Intersection with ground
        Ccw_rc = C_cw * r_c
        alpha = -self.d_c / Ccw_rc[2, 0]

        # Final symbolic expression for r_w
        self.r_w = alpha * Ccw_rc + Matrix([[self.n_c], [self.e_c], [self.d_c]])
        self.r_w = simplify(self.r_w)
        self.r_w_subs = self.r_w.subs(self.exprB, self.B).subs(self.exprC, self.C).subs(self.exprA, self.A)

    def localisationError(self):        
        r_w_bias = self.r_w.subs(self.f, self.f + self.mu_f)
        r_w_bias = r_w_bias.subs(self.x_i, self.x_i + self.mu_xi)
        r_w_bias = r_w_bias.subs(self.y_i, self.y_i + self.mu_yi)
        r_w_bias = r_w_bias.subs(self.psi, self.psi + self.mu_psi)
        r_w_bias = r_w_bias.subs(self.theta, self.theta + self.mu_theta)
        r_w_bias = r_w_bias.subs(self.n_c, self.n_c + self.mu_nc)
        r_w_bias = r_w_bias.subs(self.e_c, self.e_c + self.mu_ec)
        r_w_bias = r_w_bias.subs(self.d_c, self.d_c + self.mu_dc)

        err = r_w_bias - self.r_w

        # MACRO SUBSTITUTIONS FOR SIMPLIFICATION
        # Orient such that north corrsponds to the longitudinal axis and east to the lateral axis
        err = err.subs(self.psi, 0)

        # Move the object to the center of the image
        err = err.subs(self.x_i - self.c_x, 0)
        err = err.subs(self.y_i - self.c_y, 0)

        # ERROR SOURCE SPECIFIC SIMPLIFICATIONS
        # Horizontal position error
        self.err_pos = self.zeroUndesiredContributions(err, exclude=[self.mu_nc, self.mu_ec])

        # Vertical position error
        self.err_alt = self.zeroUndesiredContributions(err, exclude=[self.mu_dc])
        self.err_alt_withDist = self.useDistanceThetaSubs(self.err_alt)

        # Attitude pitch error
        self.err_pitch = self.zeroUndesiredContributions(err, exclude=[self.mu_theta])
        self.err_pitch_smallAngle = self.smallAngleApproximation(self.err_pitch, self.mu_theta)
        self.err_pitch_smallAngle_withDist = self.useDistanceThetaSubs(self.err_pitch_smallAngle)

        # Attitude yaw error
        self.err_yaw = self.zeroUndesiredContributions(err, exclude=[self.mu_psi])
        self.err_yaw_smallAngle = self.smallAngleApproximation(self.err_yaw, self.mu_psi)
        self.err_yaw_smallAngle_withDist = self.useDistanceThetaSubs(self.err_yaw_smallAngle)

        # X image coordinate error
        self.err_xi = self.zeroUndesiredContributions(err, exclude=[self.mu_xi])
        self.err_xi_withDist = self.useDistanceThetaSubs(self.err_xi)

        # Y image coordinate error
        self.err_yi = self.zeroUndesiredContributions(err, exclude=[self.mu_yi])
        self.err_yi_withDist = self.useDistanceThetaSubs(self.err_yi)
        
    def sizeError(self):
        # Define a origin object
        r_w_origin = self.r_w

        # Define an object laterally offset
        r_w_lateral = self.r_w.subs(self.c_x - self.x_i, self.c_x - self.x_i + self.B)

        # Define an object longitudinally offset
        r_w_longitudinal = self.r_w.subs(self.c_y - self.y_i, self.c_y - self.y_i + self.C)

        # Size function vector expressions
        SF_lateral = r_w_lateral - r_w_origin
        SF_longitudinal = r_w_longitudinal - r_w_origin

        # BIASED SIZE MEASUREMENT 
        # Lateral object
        SF_lateral_bias = SF_lateral.subs(self.n_c, self.n_c + self.mu_nc)
        SF_lateral_bias = SF_lateral_bias.subs(self.e_c, self.e_c + self.mu_ec)
        SF_lateral_bias = SF_lateral_bias.subs(self.d_c, self.d_c + self.mu_dc)
        SF_lateral_bias = SF_lateral_bias.subs(self.f, self.f + self.mu_f)
        SF_lateral_bias = SF_lateral_bias.subs(self.x_i, self.x_i + self.mu_xi)
        SF_lateral_bias = SF_lateral_bias.subs(self.y_i, self.y_i + self.mu_yi)
        SF_lateral_bias = SF_lateral_bias.subs(self.psi, self.psi + self.mu_psi)
        SF_lateral_bias = SF_lateral_bias.subs(self.theta, self.theta + self.mu_theta)

        # Longitudinal object
        SF_longitudinal_bias = SF_longitudinal.subs(self.n_c, self.n_c + self.mu_nc)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.e_c, self.e_c + self.mu_ec)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.d_c, self.d_c + self.mu_dc)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.f, self.f + self.mu_f)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.x_i, self.x_i + self.mu_xi)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.y_i, self.y_i + self.mu_yi)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.psi, self.psi + self.mu_psi)
        SF_longitudinal_bias = SF_longitudinal_bias.subs(self.theta, self.theta + self.mu_theta)

        
        # Horizontal size error LATERAL
        self.ErrSize_pos_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_nc, self.mu_ec], direction='lateral')

        # Horizontal size error LONGITUDINAL
        self.ErrSize__pos_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_nc, self.mu_ec], direction='longitudinal')

        # Vertical size error LATERAL
        self.ErrSize_alt_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_dc], direction='lateral')
        self.ErrSize_alt_lat_withDist = self.ErrSize_alt_lat.subs(tan(self.theta), self.d_c / self.l)

        # Vertical size error LONGITUDINAL
        self.ErrSize_alt_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_dc], direction='longitudinal')

        # Attitude pitch size error LATERAL
        self.ErrSize_pitch_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_theta], direction='lateral')
        self.ErrSize_pitch_lat_smallAngle = self.smallAngleApproximation(self.ErrSize_pitch_lat, self.mu_theta)
        self.ErrSize_pitch_lat_smallAngle_withDist = self.useDistanceThetaSubs(self.ErrSize_pitch_lat_smallAngle)

        # Attitude pitch size error LONGITUDINAL
        self.ErrSize_pitch_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_theta], direction='longitudinal')
        self.ErrSize_yaw_long_smallAngle = self.smallAngleApproximation(self.ErrSize_pitch_long, self.mu_theta)
        self.ErrSize_yaw_long_smallAngle_withDist = self.useDistanceThetaSubs(self.ErrSize_yaw_long_smallAngle)
        self.ErrSize_yaw_long_smallAngle_withDist_smallObject = simplify(self.ErrSize_yaw_long_smallAngle_withDist.subs(self.s, 0))

        # Attitude yaw size error LATERAL
        self.ErrSize_yaw_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_psi], direction='lateral', norm="bespoke")

        # Attitude yaw size error LONGITUDINAL
        self.ErrSize_yaw_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_psi], direction='longitudinal', norm="bespoke")

        # X image coordinate size error LATERAL
        self.ErrSize_xi_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_xi], direction='lateral')

        # X image coordinate size error LONGITUDINAL
        self.ErrSize_xi_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_xi], direction='longitudinal', norm="sympy")
        self.ErrSize_xi_long_withDist = self.useDistanceThetaSubs(self.ErrSize_xi_long)

        # Y image coordinate size error LATERAL
        self.ErrSize_yi_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_yi], direction='lateral')
        self.ErrSize_yi_lat_withDist = self.useDistanceThetaSubs(self.ErrSize_yi_lat)

        # Y image coordinate size error LONGITUDINAL
        self.ErrSize_yi_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_yi], direction='longitudinal', norm="sympy")
        self.ErrSize_yi_long_withDist = self.useDistanceThetaSubs(self.ErrSize_yi_long)

        # Focal length size error LATERAL
        self.ErrSize_f_lat = self.compute_SF_error(SF_lateral, SF_lateral_bias, activeError=[self.mu_f], direction='lateral')
        self.ErrSize_f_lat_withDist = self.useDistanceThetaSubs(self.ErrSize_f_lat)

        # Focal length size error LONGITUDINAL
        self.ErrSize_f_long = self.compute_SF_error(SF_longitudinal, SF_longitudinal_bias, activeError=[self.mu_f], direction='longitudinal')
        self.ErrSize_f_long_withDist = self.useDistanceThetaSubs(self.ErrSize_f_long)
        self.ErrSize_f_long_withDist_smallObject = simplify(self.ErrSize_f_long_withDist.subs(self.s, 0))

    def imageAttiEquivalence(self):
        # Pitch and Y image equivalence
        eq_vec = simplify(self.err_pitch_smallAngle_withDist[0] - self.err_yi_withDist[0])
        self.err_pitch_yi_equivalence = solve(eq_vec, self.mu_theta)[0]

        # Yaw and X image equivalence
        eq_vec = simplify(self.err_yaw_smallAngle_withDist[1] - self.err_xi_withDist[1])
        self.err_yaw_xi_equivalence = solve(eq_vec, self.mu_psi)[0]
        
        
if __name__ == "__main__":
    sym_eq = SymbolicEquations()
    sym_eq.localisation()
    sym_eq.localisationError()
    sym_eq.imageAttiEquivalence()
    sym_eq.sizeError()
    sym_eq.printAllLatex(Path("equations/symbolic_equations.tex"))